# PT-CycleGAN-for-MRI-Synthesis
Official implementation of PT-CycleGAN: Progressive Transformer-based CycleGAN for 3T to 7T-like MRI Synthesis.

This folder contains the core Python scripts to:

- Train the model (`Train.py`)
- Run reconstruction/inference (`reconstruct.py`)
- Compute image quality metrics between generated and real images (`extract_matrices.py`)

Other key modules:

- Model definitions (`Model.py`)
- Dataset definitions (`Dataset.py`, `Dataset2.py`)
- Utilities (checkpointing, seeding, etc.) (`Utils.py`)

## Prerequisites

- Python 3.10+ recommended
- (Optional) NVIDIA GPU with CUDA for faster training/inference

## 1) Create and activate a virtual environment

### Windows (PowerShell)

```powershell
cd path\to\PT_CycleGAN

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
cd path/to/PT_CycleGAN

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you want GPU-enabled PyTorch, install PyTorch according to the official instructions for your CUDA version, then install the remaining dependencies.

- PyTorch install selector: `https://pytorch.org/get-started/locally/`

## 3) Prepare your datasets (folder structure)

The code is written to be path-agnostic. You only need to set the correct folder paths in the scripts.

### Training dataset

Recommended structure:

```text
training_dataset/
  3T/
    <3T_train_images...>
  7T/
    <7T_train_images...>
```

`Train.py` uses:

- `TRAIN_DIR = "training_dataset"  # TODO: replace with your dataset root`
- `root_a = TRAIN_DIR + "/3T"`
- `root_b = TRAIN_DIR + "/7T"`

### Testing datasets

For reconstruction (3T test inputs):

```text
3T_testing_dataset/
  <3T_test_images...>
```

For metric extraction (real 7T test set):

```text
7T_testing_dataset/
  <7T_test_images...>
```

## 4) Train

From this folder:

```bash
python Train.py
```

Outputs:

- A `Results/` directory is created (if missing)
- Checkpoints are written periodically inside `Results/`
- Generated sample images are saved under:
  - `Results/Generated from 7T/`
  - `Results/Generated from 3T/`

Important variables you may want to adjust (near the bottom of `Train.py`):

- `NUM_EPOCHS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `IMAGE_HEIGHT`, `IMAGE_WIDTH`
- Loss weights: `LAMBDA_ADV`, `LAMBDA_IDENTITY`, `LAMBDA_TEXTURE`, `LAMBDA_STRUCT`, `LAMBDA_CYCLE`
- Incremental/transformer settings: `INCREMENTAL_EPOCHS`, `NUM_HEADS`, `PATCHES`, `UNFREEZE_EPOCH_NO`

The script automatically selects device as:

- CUDA if available, else CPU

## 5) Reconstruct / inference

Edit these variables at the top of `reconstruct.py`:

- `input_dir = "3T/test"  # TODO: replace with your input dataset path`
- `output_dir = "fake_dataset_output"  # TODO: replace with your desired output folder`
- `checkpoint = "path/to/your_checkpoint.pth"  # TODO: replace with your generator checkpoint`

Then run:

```bash
python reconstruct.py
```

Output:

- Generated images saved under `output_dir`

## 6) Compute metrics (PSNR / MSE / RMSE / SSIM)

Edit at the bottom of `extract_matrices.py`:

- `fake_folder = "path/to/your/fake_images"  # TODO: replace with your generated/fake image folder`
- `real_folder = "path/to/your/real_images"  # TODO: replace with your real image folder (e.g., 7T images)`
- `output_csv = "image_comparison_results.csv"`

Run:

```bash
python extract_matrices.py
```

Output:

- A CSV file (`output_csv`) with per-image metrics.

## Notes

- All scripts are written to run on either CUDA or CPU.
- Paths are intentionally kept as short local placeholders so this folder can be uploaded to GitHub and run on another machine.
