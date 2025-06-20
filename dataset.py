import reasoning_gym
from typing import List, Dict, Any
import numpy as np

def format_puzzle_grid(grid: List[List[int]]) -> str:
    """Convert a 2D grid into a formatted string representation."""
    return '\n'.join(' '.join(str(x if x != 0 else '_') for x in row) for row in grid)

def format_solution_grid(grid: List[List[int]]) -> str:
    """Convert a solution grid into a string representation."""
    return '\n'.join(' '.join(str(x) for x in row) for row in grid)

def generate_dataset(size: int = 200, min_empty: int = 3, max_empty: int = 8, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a formatted dataset of mini Sudoku puzzles."""
    # Generate raw puzzles using reasoning_gym
    raw_data = reasoning_gym.create_dataset(
        'mini_sudoku',
        min_empty=min_empty,
        max_empty=max_empty,
        seed=seed,
        size=size
    )

    formatted_data = []
    for item in raw_data:
        puzzle = item['metadata']['puzzle']
        solution = item['metadata']['solution']

        formatted_item = {
            'question': format_puzzle_grid(puzzle),
            'answer': format_solution_grid(solution)
        }
        formatted_data.append(formatted_item)

    return formatted_data

# Dataset generation and saving
def save_dataset(size: int = 200, train_split: float = 0.8, batch_size: int = 50, output_dir: str = "sudoku_dataset"):
    """
    Generate and save a large dataset, split into train and validation sets.

    Args:
        size: Total number of puzzles to generate
        train_split: Fraction of data to use for training
        batch_size: Number of puzzles to generate per batch
        output_dir: Directory to save the dataset
    """
    import os
    import json
    from tqdm import tqdm

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate dataset with progress bar
    print(f"Generating {size} puzzles...")
    dataset = []

    num_batches = size // batch_size + (1 if size % batch_size > 0 else 0)

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, size - i * batch_size)
        if current_batch_size <= 0:
            break
        batch = generate_dataset(
            size=current_batch_size,
            min_empty=8,
            max_empty=12,
            seed=42 + i * 1000 
        )
        dataset.extend(batch)

    # Split into train and validation sets
    train_size = int(len(dataset) * train_split)
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    # Save datasets
    print(f"\nSaving datasets to {output_dir}...")
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\nDataset statistics:")
    print(f"Total puzzles: {len(dataset)}")
    print(f"Training puzzles: {len(train_data)}")
    print(f"Validation puzzles: {len(val_data)}")

if __name__ == "__main__":
    # Only change these as needed.
    total_size = 2000    # Total puzzles to generate
    train_split = 0.9    # Train/validation split
    batch_size = 50      # Number of puzzles per batch

    save_dataset(
        size=total_size,
        train_split=train_split,
        batch_size=batch_size,
        output_dir="data"
    )
