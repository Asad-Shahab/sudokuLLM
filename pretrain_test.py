from unsloth import FastLanguageModel, PatchFastRL
from vllm import SamplingParams

def load_base_model(max_seq_length: int = 2048, lora_rank: int = 64):
    """Initialize and load the base model without trained weights."""
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

    return model, tokenizer

def solve_sudoku_base(puzzle: str, model, tokenizer):
    """
    Solve a 4x4 Sudoku puzzle using the base (untrained) model.

    Args:
        puzzle: String representation of the puzzle.
        model: Loaded base model.
        tokenizer: Loaded tokenizer.

    Returns:
        str: Model's raw response.
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

    # Generate solution using base model
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=None,  # No LoRA weights loaded
    )[0].outputs[0].text

    return output

def main():
    """Test the base model before training."""
    print("Loading base model (no training)...")
    model, tokenizer = load_base_model()

    while True:
        print("\nEnter a 4x4 Sudoku puzzle (use underscores for empty cells)")
        print("Format example:\n1 3 4 2\n2 4 3 1\n3 2 _ 4\n_ 1 2 3")
        print("Enter 'quit' to exit")

        # Get puzzle input
        lines = []
        for i in range(4):
            line = input(f"Row {i+1}: ")
            if line.lower() == 'quit':
                return
            lines.append(line)

        puzzle = '\n'.join(lines)

        # Solve puzzle using base model
        print("\nAttempting to solve puzzle with base model...")
        solution = solve_sudoku_base(puzzle, model, tokenizer)
        print("\nModel's Response:")
        print(solution)

if __name__ == "__main__":
    main()
