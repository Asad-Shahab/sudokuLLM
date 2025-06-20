# Mini Sudoku Solver - Fine-tuning with GRPO

This repository contains code to fine-tune Google's Gemma-3-4B-IT language model to solve 4x4 Mini Sudoku puzzles using GRPO (Generative Reinforcement Policy Optimization). The project demonstrates how to train a model to:

1. Follow a specific output format with reasoning and solution tags
2. Apply logical reasoning to solve Sudoku puzzles step-by-step
3. Output valid 4x4 Sudoku solutions with high accuracy

## 🚀 Setup and Installation

### Prerequisites

This project uses [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning.

**Important**: Please visit the [Unsloth installation guide](https://github.com/unslothai/unsloth?tab=readme-ov-file#-install-unsloth) to install according to your system requirements. They provide multiple installation options:
- Conda
- Pip
- Different CUDA versions
- CPU-only installations

The installation method depends on your GPU, CUDA version, and system configuration.

### Clone Repository

```bash
git clone https://github.com/Asad-Shahab/sudokuLLM.git
cd sudokuLLM
```

### Additional Dependencies

Since most dependencies are handled by the Unsloth conda installation, you only need to install a few additional packages:

```bash
pip install wandb  # Optional: for experiment tracking
```

## 📊 Dataset

This project uses the [asadshahab/mini-sudoku](https://huggingface.co/datasets/asadshahab/mini-sudoku) dataset from Hugging Face. This dataset was generated using the included `dataset.py` script and contains:
- Pre-generated 4x4 Mini Sudoku puzzles
- Puzzles with varying difficulty (different numbers of empty cells)
- Training and validation splits

The dataset format uses underscores (`_`) for empty cells:
```
2 _ 3 1
1 3 _ 4
3 1 4 2
_ _ 1 _
```

### Generate Your Own Dataset

You can generate a custom dataset using the same `dataset.py` script that was used to create the HuggingFace dataset:

```bash
python dataset.py
```

This will:
- Generate unique 4x4 Mini Sudoku puzzles using the `reasoning_gym` library
- Create puzzles with customizable difficulty (empty cells)
- Split into training (90%) and validation (10%) sets
- Save to the `data` directory in JSON format

To customize the dataset generation, modify these parameters in `dataset.py`:
- `total_size`: Number of puzzles to generate (default: 2000)
- `train_split`: Training/validation split ratio (default: 0.9)
- `min_empty`/`max_empty`: Range of empty cells per puzzle (default: 8-12)

## 🎯 Model and Training Approach

### Model
- **Base Model**: `unsloth/gemma-3-4b-it` (Google's Gemma 3 4B Instruct)
- **Training Method**: Full fine-tuning (with LoRA option available)
- **Optimization**: GRPO with multiple reward functions

### Output Format
The model is trained to output solutions in this format:
```
<start_working_out>
[Step-by-step reasoning here]
<end_working_out>
<SOLUTION>
3 1 4 2
2 4 1 3
1 2 3 4
4 3 2 1
</SOLUTION>
```

### Reward Functions
The training uses four reward functions to ensure high-quality outputs:

1. **Format Match** (2.0 points): Checks for correct reasoning and solution tags
2. **Correctness** (5.0 points): Validates the Sudoku solution
3. **Valid Numbers** (0.5 points): Ensures all numbers are 1-4
4. **Grid Format** (1.0 points): Verifies proper 4x4 grid structure

**Total Maximum Score**: 8.5 points

## 🏋️ Training

### Fine-tuning Configuration
- **Learning Rate**: 5e-6 with cosine scheduler
- **Batch Size**: 4 per device with gradient accumulation of 4
- **Epochs**: 4
- **Optimizer**: AdamW (fused)
- **Training Mode**: Full fine-tuning (LoRA adapters available but commented out)

### Run Training

```bash
python finetune.py
```

You'll be prompted for a Weights & Biases API key (optional for experiment tracking). The training process will:
1. Load the Gemma model
2. Process the dataset with the appropriate prompt format
3. Train using GRPO with the reward functions
4. Save the model to `mini-sudoku-solver/`

## 🔮 Inference

### Interactive Solver

```bash
python inference.py
```

The inference script offers multiple modes:
1. **Batch test**: Test on pre-defined puzzles
2. **Interactive mode**: Enter custom puzzles
3. **Single test**: Quick test with one puzzle
4. **Debug mode**: Analyze model outputs
5. **Test with known answer**: Compare against correct solutions
6. **Score pasted output**: Evaluate any model output

## 📈 Performance

*Performance metrics and evaluation results will be added after comprehensive testing.*

## 🛠️ Customization

### Using LoRA Instead of Full Fine-tuning

To use LoRA adapters (for lower memory usage), modify `finetune.py`:
1. Set `full_finetuning=False` in model loading
2. Uncomment the `FastModel.get_peft_model` section
3. Adjust LoRA parameters (r, alpha, dropout) as needed

### Adjusting Training Parameters

Key parameters in `finetune.py`:
- `max_seq_length`: Maximum sequence length (default: 1024)
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per GPU
- `learning_rate`: Initial learning rate

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM fine-tuning
- Google for the Gemma model family
- Hugging Face for hosting the dataset and model hub
- TRL library for GRPO implementation
- Weights & Biases for experiment tracking
