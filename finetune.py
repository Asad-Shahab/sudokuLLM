import os
import json
import re
import numpy as np
from datasets import Dataset
import torch
import wandb
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Patch GRPO before all functions
PatchFastRL("GRPO", FastLanguageModel)

def load_sudoku_dataset(file_path: str) -> Dataset:
    """Load Sudoku dataset from JSON and convert to HuggingFace Dataset format."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    formatted_data = {
        'prompt': [item['prompt'] for item in data],
        'answer': [item['answer'] for item in data]
    }

    return Dataset.from_dict(formatted_data)

def extract_grid_from_answer(answer_text: str) -> list[list[int]]:
    """Extract the grid from the answer section of the response."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_text, re.DOTALL)
    if not match:
        return None

    grid_str = match.group(1)
    try:
        grid = [[int(num) for num in row.split()] for row in grid_str.strip().split('\n')]
        return grid
    except:
        return None

def is_valid_sudoku_solution(grid: list[list[int]]) -> bool:
    """Check if a 4x4 grid is a valid Sudoku solution."""
    if not grid or len(grid) != 4 or any(len(row) != 4 for row in grid):
        return False

    grid = np.array(grid)

    # Check all numbers are 1-4
    if not all(num in [1, 2, 3, 4] for num in grid.flatten()):
        return False

    # Check rows and columns
    for i in range(4):
        if len(set(grid[i, :])) != 4 or len(set(grid[:, i])) != 4:
            return False

    # Check 2x2 boxes
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            if len(set(grid[i:i+2, j:j+2].flatten())) != 4:
                return False

    return True

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the Sudoku solution is correct."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response, correct_answer in zip(responses, answer):
        predicted_grid = extract_grid_from_answer(response)
        correct_grid = extract_grid_from_answer(correct_answer)

        if predicted_grid is None or correct_grid is None:
            rewards.append(0.0)
            continue

        if (predicted_grid == correct_grid and
            is_valid_sudoku_solution(predicted_grid)):
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if all numbers in the solution are 1-4."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    for response in responses:
        grid = extract_grid_from_answer(response)
        if grid is None:
            rewards.append(0.0)
            continue

        try:
            if all(all(num in [1, 2, 3, 4] for num in row) for row in grid):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)

    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """More lenient reward function for XML format checking."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def count_xml(text) -> float:
    """Count reward for XML tag presence and placement."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for XML formatting details."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def setup_model(max_seq_length: int = 2048, lora_rank: int = 64):
    """Initialize and setup the model with given parameters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True, # False for LoRA 16bit
        fast_inference=True, # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer

def setup_training_args():
    """Setup and return GRPO training configuration."""
    return GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8, # Decrease if out of memory
        max_prompt_length=256,
        max_completion_length=1024,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="wandb",
        run_name="sudoku-grpo-training",
        output_dir="outputs",
    )

def train_model(model, tokenizer, dataset, training_args):
    """Initialize trainer and start training."""
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    wandb.init(
        project="sudoku-grpo",
        name="training-run-1",
        config={
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "max_seq_length": 2048,
            "lora_rank": 64,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "num_generations": training_args.num_generations,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        }
    )
    
    trainer.train()
    wandb.finish()
    
    # Save the trained model
    model.save_lora("grpo_saved_lora")

def test_model(model, tokenizer, use_lora=True):
    """Test the model with a sample puzzle."""
    system_prompt = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    test_puzzle = "Solve this 4x4 Mini Sudoku puzzle:\n_ 3 1 2\n2 _ _ 4\n3 _ _ 1\n_ _ 4 _"
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_puzzle}
    ], tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
    )
    
    lora_request = model.load_lora("grpo_saved_lora") if use_lora else None
    
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text
    
    return output

def main():
    # Set up wandb
    wandb.login()
    
    # Load dataset
    dataset = load_sudoku_dataset("sudoku_dataset/train.json")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model()   

    # Run initial test before training
    print("\n=== Pre-training Test ===")
    print("Testing base model before any training:")
    pretraining_result = test_model(model, tokenizer, use_lora=False)
    print(pretraining_result)

    # Setup training arguments
    training_args = setup_training_args()
    
    print("\n=== Starting Training ===")
    # Train the model
    train_model(model, tokenizer, dataset, training_args)

    print("\n=== Post-training Tests ===")
    print("Testing model without LoRA (should be similar to pre-training):")
    print(test_model(model, tokenizer, use_lora=False))
    
    print("\nTesting model with trained LoRA weights:")
    print(test_model(model, tokenizer, use_lora=True))
        
    # Save results to file for comparison
    with open("model_comparison_results.txt", "w") as f:
        f.write("=== Pre-training Output ===\n")
        f.write(pretraining_result)
        f.write("\n\n=== Post-training Output (with LoRA) ===\n")
        f.write(test_model(model, tokenizer, use_lora=True))


if __name__ == "__main__":
    main()