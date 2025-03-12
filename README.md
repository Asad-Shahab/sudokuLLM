# Sudoku Finetuning

This repository contains code to finetune a language model (Qwen/Qwen2.5-3B-Instruct) to solve 4x4 Mini Sudoku puzzles with GRPO (Generative Reinforcement Policy Optimization). The project demonstrates how to train a model to:

1. Follow a specific XML output format
2. Apply logical reasoning to solve Sudoku puzzles
3. Output valid 4x4 Sudoku solutions


## Setup and Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Asad-Shahab/sudoku_finetuning.git
cd sudoku_finetuning
pip install -r requirements.txt
```

## Dataset Generation

The project uses a custom dataset of 4x4 Mini Sudoku puzzles. You can generate the dataset with:

```bash
python dataset.py
```

This will:
- Generate 1000 unique 4x4 Mini Sudoku puzzles
- Split them into training (90%) and validation (10%) sets
- Save the formatted data in the `sudoku_dataset` directory

To verify the dataset's integrity:

```bash
python verify_dataset.py
```

## Dataset Format

Each example in the dataset follows this structure:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n"
    },
    {
      "role": "user",
      "content": "Solve this 4x4 Mini Sudoku puzzle:\n3 1 4 2\n_ 4 1 3\n1 2 _ _\n4 3 2 1"
    }
  ],
  "answer": "<reasoning>  </reasoning>\n<answer>\n3 1 4 2\n2 4 1 3\n1 2 3 4\n4 3 2 1\n</answer>"
}
```

## Finetuning

The finetuning process uses GRPO with several reward functions to train the model:

1. `correctness_reward_func`: Rewards correct Sudoku solutions
2. `int_reward_func`: Checks if all numbers in the solution are valid (1-4)
3. `strict_format_reward_func` and `soft_format_reward_func`: Verify XML formatting
4. `xmlcount_reward_func`: Rewards proper XML tag placement

To run the finetuning:

```bash
python finetune.py
```

For long training sessions, you can use tmux:

```bash
tmux new -s sudoku_training
python finetune.py
# Detach with Ctrl+b, then d
# Reconnect later with: tmux attach -t sudoku_training
```

## Inference

To run inference with the finetuned model:

```bash
python inference.py
```

This will load the trained model and allow you to enter 4x4 Sudoku puzzles for the model to solve interactively.

To test the base model performance (without finetuning):

```bash
python pretrain_test.py
```


## Acknowledgments

- This project uses the [unsloth](https://github.com/unsloth/unsloth) library for efficient finetuning
- TRL (Transformer Reinforcement Learning) library for GRPO implementation
- Weights & Biases for experiment tracking
