from unsloth import FastModel
import torch
import re
from transformers import TextStreamer

# Helper functions
def extract_grid_from_answer(text):
    """Extract a 4x4 grid from the model's answer text."""
    if text is None:
        return None
        
    try:
        lines = []
        for line in text.strip().split('\n'):
            if re.search(r'[1-4]', line):
                numbers = [int(n) for n in re.findall(r'[1-4]', line)]
                if len(numbers) == 4:
                    lines.append(numbers)
        
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

def extract_solution(text):
    """Extract solution from model output."""
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
    # This helps when model doesn't use exact tags
    lines = text.strip().split('\n')
    potential_grid_lines = []
    
    # Look for lines with exactly 4 numbers from 1-4
    for line in reversed(lines):  # Start from the end
        if re.match(r'^\s*[1-4](\s+[1-4]){3}\s*$', line.strip()):
            potential_grid_lines.insert(0, line.strip())
            if len(potential_grid_lines) == 4:
                return '\n'.join(potential_grid_lines)
        elif potential_grid_lines:
            # If we've started finding grid lines but hit a non-grid line, stop
            break
    
    return None

def format_grid(grid):
    """Format grid for pretty printing."""
    if grid is None:
        return "Invalid grid"
    return '\n'.join([' '.join(map(str, row)) for row in grid])

# Scoring functions (adapted from training)
def score_format_match(response):
    """Check if output follows the required format. Max: 2.0"""
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    
    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]*"
        rf"(?:<end_of_turn>)?"
        rf"[\s]*$",
        flags=re.MULTILINE | re.DOTALL
    )
    return 2.0 if match_format.search(response) is not None else 0.0

def score_correctness(predicted_grid, correct_answer=None):
    """Check if the Sudoku solution is correct. Max: 5.0"""
    if predicted_grid is None:
        return 0.0
    
    # If we have a correct answer, compare
    if correct_answer:
        correct_grid = extract_grid_from_answer(correct_answer)
        if correct_grid and predicted_grid == correct_grid and is_valid_sudoku_solution(predicted_grid):
            return 5.0
    # Otherwise just check validity
    elif is_valid_sudoku_solution(predicted_grid):
        return 5.0
    
    return 0.0

def score_valid_numbers(grid):
    """Check if all numbers in the solution are 1-4. Max: 0.5"""
    if grid is None:
        return 0.0
    
    try:
        if all(all(num in [1, 2, 3, 4] for num in row) for row in grid):
            return 0.5
        else:
            return 0.0
    except:
        return 0.0

def score_grid_format(solution_text):
    """Check the format of the grid output. Max: 1.0"""
    if solution_text is None:
        return 0.0
    
    lines = solution_text.strip().split('\n')
    valid_lines = 0
    
    for line in lines:
        if re.match(r'^\s*[1-4](\s+[1-4]){3}\s*$', line):
            valid_lines += 1
    
    if valid_lines == 4:
        return 1.0
    elif valid_lines > 0:
        return valid_lines / 8.0
    else:
        return 0.0

def calculate_all_scores(model_output, solution_text, grid, correct_answer=None):
    """Calculate all scores for the model output."""
    scores = {
        "format_match": score_format_match(model_output),
        "correctness": score_correctness(grid, correct_answer),
        "valid_numbers": score_valid_numbers(grid),
        "grid_format": score_grid_format(solution_text),
    }
    scores["total"] = sum(scores.values())
    scores["max_possible"] = 8.5  # 2.0 + 5.0 + 0.5 + 1.0
    return scores

def parse_sudoku_input(puzzle_str):
    """Parse sudoku input, handling various formats."""
    # Replace common empty cell indicators with underscore
    puzzle_str = puzzle_str.replace('*', '_').replace('.', '_').replace('0', '_')
    # Then replace underscore with space for the model
    return puzzle_str.replace('_', ' ')

# Load the fine-tuned model
print("Loading fine-tuned model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, tokenizer = FastModel.from_pretrained(
    model_name="mini-sudoku-solver",  # Your saved model directory
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_8bit=False,
)

# System prompt (same as training)
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

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

def solve_sudoku(puzzle_input, temperature=0.7, show_reasoning=True, show_scores=True, correct_answer=None):
    """Solve a single sudoku puzzle."""
    # Parse input
    parsed_puzzle = parse_sudoku_input(puzzle_input)
    
    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parsed_puzzle},
    ]
    
    # Tokenize
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Generate
    print(f"\nInput puzzle:\n{puzzle_input}\n")
    print("Generating solution...\n")
    
    if show_reasoning:
        # Stream output
        outputs = model.generate(
            **tokenizer(text, return_tensors="pt").to(device),
            max_new_tokens=2048,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
    else:
        # No streaming
        outputs = model.generate(
            **tokenizer(text, return_tensors="pt").to(device),
            max_new_tokens=2048,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
        )
    
    # Extract and validate solution
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # More robust extraction: find where the model's output starts
    # Look for the end of the user message and take everything after
    user_message_end = parsed_puzzle
    if user_message_end in generated_text:
        model_output = generated_text.split(user_message_end)[-1]
    else:
        # Fallback: take the last part of the text
        model_output = generated_text
    
    solution_text = extract_solution(model_output)
    
    result = {
        "grid": None,
        "valid": False,
        "scores": None
    }
    
    if solution_text:
        grid = extract_grid_from_answer(solution_text)
        if grid:
            print(f"\nExtracted solution:\n{format_grid(grid)}")
            is_valid = is_valid_sudoku_solution(grid)
            
            if is_valid:
                print("\n✓ Solution is valid!")
            else:
                print("\n✗ Solution is invalid!")
                # Debug info
                print(f"Debug - Grid extracted: {grid}")
            
            result["grid"] = grid
            result["valid"] = is_valid
            
            # Calculate and display scores
            if show_scores:
                scores = calculate_all_scores(model_output, solution_text, grid, correct_answer)
                result["scores"] = scores
                
                print("\n" + "="*40)
                print("SCORING BREAKDOWN:")
                print("="*40)
                print(f"Format Match (reasoning + solution tags): {scores['format_match']:.1f}/2.0")
                print(f"Correctness (valid sudoku solution):      {scores['correctness']:.1f}/5.0")
                print(f"Valid Numbers (all numbers are 1-4):      {scores['valid_numbers']:.1f}/0.5")
                print(f"Grid Format (4x4 grid structure):         {scores['grid_format']:.1f}/1.0")
                print("-"*40)
                print(f"TOTAL SCORE:                              {scores['total']:.1f}/{scores['max_possible']}")
                print(f"Percentage:                               {(scores['total']/scores['max_possible']*100):.1f}%")
                print("="*40)
                
        else:
            print("\n✗ Could not extract valid grid from solution.")
    else:
        print("\n✗ Could not extract valid solution from output.")
        # Debug info
        print(f"Debug - Solution text found: {solution_text}")
        print(f"Debug - Last 500 chars of output:\n{model_output[-500:]}")
        
        # Still calculate format score
        if show_scores:
            scores = calculate_all_scores(model_output, None, None, correct_answer)
            result["scores"] = scores
            print("\n" + "="*40)
            print("SCORING BREAKDOWN (Failed extraction):")
            print("="*40)
            print(f"Format Match: {scores['format_match']:.1f}/2.0")
            print("Other scores: 0.0 (no valid solution found)")
            print("="*40)
    
    return result

# Example puzzles to test
test_puzzles = [
    "1 3 _ 4\n2 4 1 3\n4 2 3 _\n3 1 4 2",  # Only 2 missing cells
    "_ 2 _ 4\n4 _ _ _\n_ _ _ 3\n3 _ 1 _",  # More complex
    "2 _ 3 1\n1 3 4 2\n3 2 1 4\n4 _ _ 3",  # 2 missing cells
]

def run_batch_test():
    """Test multiple puzzles."""
    print("Running batch test on example puzzles...\n")
    print("="*50)
    
    results = []
    all_scores = []
    
    for i, puzzle in enumerate(test_puzzles, 1):
        print(f"\nPuzzle {i}:")
        result = solve_sudoku(puzzle, temperature=0.3, show_reasoning=True, show_scores=True)
        results.append(result["valid"])
        if result["scores"]:
            all_scores.append(result["scores"]["total"])
        print("="*50)
    
    print(f"\nBatch test complete: {sum(results)}/{len(results)} solved successfully")
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"Average score: {avg_score:.1f}/8.5 ({avg_score/8.5*100:.1f}%)")

def run_interactive():
    """Interactive mode for custom puzzles."""
    print("\nInteractive Mini-Sudoku Solver")
    print("Enter puzzles using _ for empty cells (or use *, ., or 0)")
    print("Example: 4 _ _ _")
    print("Type 'quit' to exit\n")
    
    while True:
        print("\nEnter your 4x4 puzzle (4 lines):")
        lines = []
        for i in range(4):
            line = input(f"Row {i+1}: ")
            if line.lower() == 'quit':
                print("Goodbye!")
                return
            lines.append(line)
        
        puzzle = '\n'.join(lines)
        
        # Ask for temperature
        temp_input = input("\nTemperature (0.1-1.0, default 0.7): ").strip()
        temperature = 0.7
        if temp_input:
            try:
                temperature = float(temp_input)
                temperature = max(0.1, min(1.0, temperature))
            except:
                print("Using default temperature 0.7")
        
        # Ask if they want to see scores
        show_scores_input = input("Show scoring breakdown? (y/n, default y): ").strip().lower()
        show_scores = show_scores_input != 'n'
        
        result = solve_sudoku(puzzle, temperature=temperature, show_reasoning=True, show_scores=show_scores)
        
        if input("\nSolve another puzzle? (y/n): ").lower() != 'y':
            break

# Main execution
if __name__ == "__main__":
    print("\nMini-Sudoku Inference Tool")
    print("\nScoring System (Total: 8.5 points):")
    print("- Format Match (2.0): Uses correct reasoning and solution tags")
    print("- Correctness (5.0): Valid sudoku solution")
    print("- Valid Numbers (0.5): All numbers are 1-4")
    print("- Grid Format (1.0): Proper 4x4 grid structure")
    print("\nOptions:")
    print("1. Run batch test")
    print("2. Interactive mode")
    print("3. Single test")
    print("4. Debug extraction test")
    print("5. Test with known answer")
    print("6. Score a pasted model answer")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        run_batch_test()
    elif choice == "2":
        run_interactive()
    elif choice == "4":
        # Debug mode to test extraction
        print("\nDebug mode - paste the model's output:")
        print("(Press Enter twice when done)")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        test_output = '\n'.join(lines[:-1])  # Remove last empty line
        solution = extract_solution(test_output)
        print(f"\nExtracted solution text:\n{solution}")
        if solution:
            grid = extract_grid_from_answer(solution)
            print(f"\nExtracted grid: {grid}")
            if grid:
                print(f"\nFormatted:\n{format_grid(grid)}")
                print(f"\nValid: {is_valid_sudoku_solution(grid)}")
                # Calculate scores
                scores = calculate_all_scores(test_output, solution, grid)
                print("\n" + "="*40)
                print("SCORING BREAKDOWN:")
                print("="*40)
                print(f"Format Match: {scores['format_match']:.1f}/2.0")
                print(f"Correctness: {scores['correctness']:.1f}/5.0")
                print(f"Valid Numbers: {scores['valid_numbers']:.1f}/0.5")
                print(f"Grid Format: {scores['grid_format']:.1f}/1.0")
                print("-"*40)
                print(f"TOTAL: {scores['total']:.1f}/8.5 ({scores['total']/8.5*100:.1f}%)")
                print("="*40)
    elif choice == "5":
        # Test with known answer
        puzzle = "1 3 _ 4\n2 4 1 3\n4 2 3 _\n3 1 4 2"
        correct_answer = "1 3 2 4\n2 4 1 3\n4 2 3 1\n3 1 4 2"
        print(f"\nTesting with known answer...")
        print(f"Correct answer:\n{correct_answer}\n")
        result = solve_sudoku(puzzle, temperature=0.3, show_reasoning=True, 
                            show_scores=True, correct_answer=correct_answer)
    elif choice == "6":
        print("\nPaste the complete model answer to score (Press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        model_output = '\n'.join(lines[:-1])  # Remove last empty line
        solution = extract_solution(model_output)
        grid = extract_grid_from_answer(solution) if solution else None
        scores = calculate_all_scores(model_output, solution, grid)
        print("\n" + "="*40)
        print("SCORING BREAKDOWN:")
        print("="*40)
        print(f"Format Match: {scores['format_match']:.1f}/2.0")
        print(f"Correctness: {scores['correctness']:.1f}/5.0")
        print(f"Valid Numbers: {scores['valid_numbers']:.1f}/0.5")
        print(f"Grid Format: {scores['grid_format']:.1f}/1.0")
        print("-"*40)
        print(f"TOTAL: {scores['total']:.1f}/8.5 ({scores['total']/8.5*100:.1f}%)")
        print("="*40)
    else:
        # Default single test
        puzzle = "2 _ 3 1\n1 3 4 2\n3 2 1 4\n4 _ _ 3"
        result = solve_sudoku(puzzle, temperature=0.7, show_reasoning=True, show_scores=True)
