from unsloth import FastModel
import os
import re
import numpy as np
import wandb
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer

# Ask for wandb API key in terminal
wandb_api_key = input("Enter your Weights & Biases API key (press Enter to skip wandb integration): ")
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
    use_wandb = True
    wandb.init(project="mini-sudoku-solver", name="gemma-mini-sudoku-3")
else:
    print("Skipping wandb integration")
    use_wandb = False

# Helper functions for Sudoku validation
def extract_grid_from_answer(text):
    """Extract a 4x4 grid from the model's answer text."""
    if text is None:
        return None
        
    try:
        # Find numeric grid (assumes format like "1 2 3 4\n2 3 4 1\n...")
        lines = []
        for line in text.strip().split('\n'):
            # Filter only lines with numbers
            if re.search(r'[1-4]', line):
                # Extract numbers from the line
                numbers = [int(n) for n in re.findall(r'[1-4]', line)]
                if len(numbers) == 4:  # Ensure we have exactly 4 numbers
                    lines.append(numbers)
        
        # Check if we have a complete 4x4 grid
        if len(lines) == 4 and all(len(line) == 4 for line in lines):
            return lines
        return None
    except Exception:
        return None

def is_valid_sudoku_solution(grid):
    """Check if a 4x4 Sudoku solution is valid."""
    if grid is None or len(grid) != 4 or any(len(row) != 4 for row in grid):
        return False
    
    # Check rows
    for row in grid:
        if sorted(row) != [1, 2, 3, 4]:
            return False
    
    # Check columns
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if sorted(column) != [1, 2, 3, 4]:
            return False
    
    # Check 2x2 sub-grids
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            sub_grid = []
            for r in range(box_row, box_row + 2):
                for c in range(box_col, box_col + 2):
                    sub_grid.append(grid[r][c])
            if sorted(sub_grid) != [1, 2, 3, 4]:
                return False
    
    return True

def parse_sudoku_question(question):
    """Parse the input sudoku question format."""
    # Only handle underscore replacement since that's what the dataset uses
    question = question.replace('_', ' ')
    return question

# Define reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Reward function that checks if the Sudoku solution is correct. Max: 5.0"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, correct_answer in zip(responses, answer):
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue
            
        predicted_grid = extract_grid_from_answer(solution_text)
        correct_grid = extract_grid_from_answer(correct_answer)
        
        if predicted_grid is None or correct_grid is None:
            rewards.append(0.0)
            continue
            
        if predicted_grid == correct_grid and is_valid_sudoku_solution(predicted_grid):
            rewards.append(5.0)
        else:
            rewards.append(0.0)
            
    return rewards

def int_reward_func(completions, **kwargs):
    """Reward function that checks if all numbers in the solution are 1-4. Max: 0.5"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response in responses:
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue
            
        grid = extract_grid_from_answer(solution_text)
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

def grid_format_reward_func(completions, **kwargs):
    """Reward function for checking the format of the output. Max: 1.0"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response in responses:
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue
            
        # Check if the solution has a grid-like structure
        lines = solution_text.strip().split('\n')
        valid_lines = 0
        
        for line in lines:
            # Check if line contains 4 numbers (1-4) with possible spaces
            if re.match(r'^\s*[1-4](\s+[1-4]){3}\s*$', line):
                valid_lines += 1
                
        # Give reward based on how many valid lines we found
        if valid_lines == 4:
            rewards.append(1.0)
        elif valid_lines > 0:
            rewards.append(valid_lines / 8.0)  # Partial reward
        else:
            rewards.append(0.0)
            
    return rewards

# Create a regex format to match the reasoning and solution sections
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Function to check if the output follows the format
def match_format_exactly(completions, **kwargs):
    """Check if output follows the required format. Max: 2.0"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Clean up any end_of_turn markers
        response = response.replace("<end_of_turn>", "")
        # Try to find solution tags
        match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{reasoning_start}.+?{reasoning_end}.*?"
            rf"{solution_start}(.+?){solution_end}"
            rf"[\s]*"
            rf"(?:<end_of_turn>)?"
            rf"[\s]*$",
            flags=re.MULTILINE | re.DOTALL
        )
        if match_format.search(response) is not None:
            score += 2.0
        scores.append(score)
    return scores

# Function to extract the solution for evaluation
def extract_solution(text):
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    # Clean up any end_of_turn markers
    text = text.replace("<end_of_turn>", "")
    # Try to find solution tags
    match_format = re.compile(
        rf"{solution_start}(.+?){solution_end}",
        flags=re.MULTILINE | re.DOTALL
    )
    match = match_format.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: look for a 4x4 grid pattern at the end of the text
    lines = text.strip().split('\n')
    potential_grid_lines = []
    for line in reversed(lines):  # Start from the end
        if re.match(r'^\s*[1-4](\s+[1-4]){3}\s*$', line.strip()):
            potential_grid_lines.insert(0, line.strip())
            if len(potential_grid_lines) == 4:
                return '\n'.join(potential_grid_lines)
        elif potential_grid_lines:
            break
    return None

# Define system prompt
system_prompt = f"""You are a mini-Sudoku solving assistant. 
You will be given a 4x4 Sudoku puzzle where some cells are filled and others are empty (shown as spaces).
The goal is to fill each empty cell with a number from 1 to 4 such that:
- Each row contains all numbers from 1 to 4 exactly once
- Each column contains all numbers from 1 to 4 exactly once
- Each 2x2 sub-grid contains all numbers from 1 to 4 exactly once

Think through the solution step by step.
Place your reasoning between {reasoning_start} and {reasoning_end}.
Then, provide your complete 4x4 solution grid between {solution_start} and {solution_end}

The solution should be formatted as a 4x4 grid with spaces between numbers and newlines between rows.
For example:
{solution_start}
1 2 3 4
3 4 1 2
2 1 4 3
4 3 2 1
{solution_end}
"""

# Load and preprocess the dataset
print("Loading dataset...")
train_dataset = load_dataset("asadshahab/mini-sudoku", split="train")
val_dataset = load_dataset("asadshahab/mini-sudoku", split="validation")
print(f"Training dataset loaded with {len(train_dataset)} examples")
print(f"Validation dataset loaded with {len(val_dataset)} examples")

# Check the first example to understand structure
print("\nExample question:", train_dataset[0]["question"])
print("Example answer:", train_dataset[0]["answer"])

# Preprocess dataset
def preprocess_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parse_sudoku_question(example["question"])},
        ],
        "answer": example["answer"],
    }

processed_train_dataset = train_dataset.map(preprocess_dataset)
processed_val_dataset = val_dataset.map(preprocess_dataset)

# Model loading and training
print("\nLoading model...")
max_seq_length = 1024

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model using Unsloth with full finetuning enabled
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=True,
)

# DO NOT add LoRA adapters when doing full fine-tuning!
# Comment out or remove the FastModel.get_peft_model section

# If using LoRA (full_finetuning=False), uncomment this:
# model = FastModel.get_peft_model(
#     model,
#     finetune_vision_layers=False,
#     finetune_language_layers=True,
#     finetune_attention_modules=True,
#     finetune_mlp_modules=True,
#     r=8,
#     lora_alpha=8,
#     lora_dropout=0,
#     bias="none",
#     random_state=3407,
# )

max_prompt_length = 256
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=4,
    save_steps=100,
    report_to="wandb" if use_wandb else "none",
    output_dir="outputs",
)


# Initialize GRPO Trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[
        match_format_exactly,       # 2.0 max - format check
        correctness_reward_func,    # 5.0 max - correctness check
        int_reward_func,            # 0.5 max - valid numbers check
        grid_format_reward_func,    # 1.0 max - grid format check
    ],
    args=training_args,
    train_dataset=processed_train_dataset,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Save the model
print("\nSaving model...")
model.save_pretrained("mini-sudoku-solver")
tokenizer.save_pretrained("mini-sudoku-solver")

# Test the model with an example from the dataset format
print("\nTesting model with an example...")
test_puzzle = "2 _ 3 1\n1 3 _ 4\n3 1 4 2\n_ _ 1 _"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": parse_sudoku_question(test_puzzle)},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

print(f"\nInput puzzle:\n{test_puzzle}")
print("\nModel output:")

outputs = model.generate(
    **tokenizer(text, return_tensors="pt").to(device),
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# Extract and validate the solution
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# More robust extraction
user_message = parse_sudoku_question(test_puzzle)
if user_message in generated_text:
    model_output = generated_text.split(user_message)[-1]
else:
    model_output = generated_text

# Clean up any end_of_turn markers
model_output = model_output.replace("<end_of_turn>", "")

solution = extract_solution(model_output)
if solution:
    grid = extract_grid_from_answer(solution)
    if grid and is_valid_sudoku_solution(grid):
        print("\n✓ Generated solution is valid!")
    else:
        print("\n✗ Generated solution is invalid or incomplete.")
        print(f"Debug - Grid extracted: {grid}")
else:
    print("\n✗ Could not extract solution from output.")
    print(f"Debug - Last 500 chars of output:\n{model_output[-500:]}")

print("\nTraining completed and model saved to 'mini-sudoku-solver' directory!")
print("\nTo upload to HuggingFace, uncomment and run the following code:")
print("""
from huggingface_hub import login
login()  # Enter your token when prompted

# Push the model to HuggingFace Hub
model.push_to_hub("YOUR_USERNAME/mini-sudoku-solver")
tokenizer.push_to_hub("YOUR_USERNAME/mini-sudoku-solver")
""")
