import json
import re
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_grid_from_string(grid_str: str) -> List[List[int]]:
    """Convert string representation of grid to 2D list of integers."""
    # Replace underscores with zeros
    grid_str = grid_str.replace('_', '0')

    # Split into rows and convert to integers
    rows = grid_str.strip().split('\n')
    return [[int(num) for num in row.split()] for row in rows]

def is_valid_sudoku_grid(grid: List[List[int]]) -> bool:
    """Check if a 4x4 grid follows Sudoku rules."""
    # Convert to numpy array for easier slicing
    grid = np.array(grid)

    # Check dimensions
    if grid.shape != (4, 4):
        return False

    # Check valid numbers
    if not all(num in [0, 1, 2, 3, 4] for num in grid.flatten()):
        return False

    # Check rows and columns (excluding zeros)
    for i in range(4):
        row = grid[i, :]
        col = grid[:, i]

        # Check non-zero numbers are unique in rows and columns
        row_nums = [x for x in row if x != 0]
        col_nums = [x for x in col if x != 0]
        if len(row_nums) != len(set(row_nums)) or len(col_nums) != len(set(col_nums)):
            return False

    # Check 2x2 boxes
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            box = grid[i:i+2, j:j+2].flatten()
            box_nums = [x for x in box if x != 0]
            if len(box_nums) != len(set(box_nums)):
                return False

    return True

def verify_dataset(dataset: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """Verify the dataset format and content."""
    stats = {
        'total_examples': len(dataset),
        'valid_format': 0,
        'valid_puzzles': 0,
        'valid_solutions': 0,
        'errors': []
    }

    for i, example in enumerate(tqdm(dataset, desc="Verifying examples")):
        try:
            # Check basic structure
            if not all(key in example for key in ['prompt', 'answer']):
                stats['errors'].append(f"Example {i}: Missing required keys")
                continue

            # Check prompt format
            prompt = example['prompt']
            if not isinstance(prompt, list) or len(prompt) != 2:
                stats['errors'].append(f"Example {i}: Invalid prompt format")
                continue

            # Check system prompt
            if prompt[0]['role'] != 'system' or '<reasoning>' not in prompt[0]['content']:
                stats['errors'].append(f"Example {i}: Invalid system prompt")
                continue

            # Check user prompt
            user_content = prompt[1]['content']
            if not user_content.startswith("Solve this 4x4 Mini Sudoku puzzle:"):
                stats['errors'].append(f"Example {i}: Invalid user prompt")
                continue

            # Extract and verify puzzle
            puzzle_str = '\n'.join(user_content.split('\n')[1:])
            puzzle = extract_grid_from_string(puzzle_str)
            if not is_valid_sudoku_grid(puzzle):
                stats['errors'].append(f"Example {i}: Invalid puzzle")
                continue

            # Check answer format
            answer = example['answer']
            if not re.match(r'<reasoning>.*</reasoning>\s*<answer>.*</answer>', answer, re.DOTALL):
                stats['errors'].append(f"Example {i}: Invalid answer XML format")
                continue

            # Extract and verify solution
            solution_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer, re.DOTALL)
            if not solution_match:
                stats['errors'].append(f"Example {i}: Cannot extract solution")
                continue

            solution = extract_grid_from_string(solution_match.group(1))
            if not is_valid_sudoku_grid(solution):
                stats['errors'].append(f"Example {i}: Invalid solution")
                continue

            # Verify solution actually solves the puzzle
            puzzle_np = np.array(puzzle)
            solution_np = np.array(solution)
            if not np.all((puzzle_np == 0) | (puzzle_np == solution_np)):
                stats['errors'].append(f"Example {i}: Solution doesn't match puzzle")
                continue

            stats['valid_format'] += 1
            stats['valid_puzzles'] += 1
            stats['valid_solutions'] += 1

        except Exception as e:
            stats['errors'].append(f"Example {i}: Unexpected error: {str(e)}")

    if verbose:
        print("\nDataset Verification Results:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Valid format: {stats['valid_format']}")
        print(f"Valid puzzles: {stats['valid_puzzles']}")
        print(f"Valid solutions: {stats['valid_solutions']}")
        print(f"Number of errors: {len(stats['errors'])}")
        if stats['errors']:
            print("\nFirst 5 errors:")
            for error in stats['errors'][:5]:
                print(error)

    return stats

if __name__ == "__main__":
    # Example usage
    print("Loading training dataset...")
    train_data = load_dataset("sudoku_dataset/train.json")
    stats = verify_dataset(train_data)

    print("\nLoading validation dataset...")
    val_data = load_dataset("sudoku_dataset/val.json")
    val_stats = verify_dataset(val_data)