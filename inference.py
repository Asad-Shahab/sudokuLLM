from unsloth import FastLanguageModel, PatchFastRL
from vllm import SamplingParams

def load_model(max_seq_length: int = 2048, lora_rank: int = 64):
    """Initialize and load the model with trained weights."""
    # Patch GRPO
    PatchFastRL("GRPO", FastLanguageModel)
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    
    # Configure model
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

def solve_sudoku(puzzle: str, model, tokenizer):
    """
    Solve a 4x4 Sudoku puzzle using the trained model.
    
    Args:
        puzzle: String representation of the puzzle (e.g., "_ _ _ _\n_ _ _ _\n_ 1 3 _\n_ 4 _ 1")
        model: Loaded model
        tokenizer: Loaded tokenizer
    
    Returns:
        str: Model's response including reasoning and solution
    """
    system_prompt = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    
    # Format the input
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Solve this 4x4 Mini Sudoku puzzle:\n{puzzle}"}
    ], tokenize=False, add_generation_prompt=True)
    
    # Set generation parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
    )
    
    # Generate solution using trained LoRA weights
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    
    return output

def main():
    # Example usage
    print("Loading model...")
    model, tokenizer = load_model()
    
    while True:
        print("\nEnter a 4x4 Sudoku puzzle (use underscores for empty cells)")
        print("Format example:\n_ _ _ _\n_ _ _ _\n_ 1 3 _\n_ 4 _ 1")
        print("Enter 'quit' to exit")
        
        # Get puzzle input
        lines = []
        for i in range(4):
            line = input(f"Row {i+1}: ")
            if line.lower() == 'quit':
                return
            lines.append(line)
        
        puzzle = '\n'.join(lines)
        
        # Solve puzzle
        print("\nSolving puzzle...")
        solution = solve_sudoku(puzzle, model, tokenizer)
        print("\nSolution:")
        print(solution)

if __name__ == "__main__":
    main()