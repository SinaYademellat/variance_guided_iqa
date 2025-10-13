# Image Quality Evaluation Tool

This document explains how to use the `aic4-eval.py` script to evaluate image quality.

## Table of Contents
- [What Does This Tool Do?](#what-does-this-tool-do)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Understanding the Feature Extractor](#understanding-the-feature-extractor)
- [Dataset Preparation](#dataset-preparation)
- [Running the Script](#running-the-script)
  - [Mode 1: Single Image Pair Evaluation](#mode-1-single-image-pair-evaluation)
  - [Mode 2: Full Dataset Evaluation](#mode-2-full-dataset-evaluation)
- [Understanding the Output](#understanding-the-output)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

---

## What Does This Tool Do?

The `aic4-eval.py` script evaluates the quality of distorted images by comparing them to their original (reference) versions. It uses a deep learning model to calculate a quality score. The higher the score, the better the quality of the distorted image.

The tool can work in two ways:
1. **Single mode**: Compare one reference image to one distorted image
2. **Dataset mode**: Compare many reference images to their corresponding distorted versions all at once

---

## Prerequisites

Before you can use this tool, you need:

1. **A computer with:**
   - Linux, macOS, or Windows operating system
   - Optional but recommended: NVIDIA GPU with CUDA support (makes processing much faster)

2. **Python installed:**
   - Python version 3.10 or higher
   - You can check by running: `python --version` or `python3 --version`

3. **Basic command line knowledge:**
   - How to open a terminal/command prompt
   - How to navigate folders using `cd` command
   - How to run Python scripts

---

## Environment Setup

Follow these steps carefully to set up your environment. Each step is important!

### Step 1: Install Python (if not already installed)

**On Linux/Ubuntu:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**On macOS:**
```bash
# Install Homebrew first if you don't have it (visit https://brew.sh)
brew install python3
```

**On Windows:**
Download and install Python from [python.org](https://www.python.org/downloads/)

### Step 2: Create a Virtual Environment

A virtual environment is like a separate workspace for this project. It keeps all the required packages isolated from other Python projects on your computer.

**Navigate to the project directory:**
```bash
cd /path/to/variance_guided_iqa
```

**Create the virtual environment:**
```bash
python3 -m venv venv
```

This creates a folder called `venv` in your project directory.

### Step 3: Activate the Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
venv\Scripts\activate
```

After activation, you should see `(venv)` at the beginning of your command line prompt. This means you're now working inside the virtual environment.

### Step 4: Install Required Packages

The script needs several Python packages to work. All of them are listed in the `requirements.txt` file.

**Install all packages at once:**
```bash
pip install -r requirements.txt
```

This will download and install:
- **PyTorch**: The deep learning framework that powers the model
- **torchvision**: Provides pre-trained models and image processing tools
- **Pillow**: For loading and handling images
- **tqdm**: Shows progress bars so you know how long processing will take
- And several other supporting packages

**Note:** This installation might take 5-15 minutes depending on your internet speed. PyTorch is a large package.

### Step 5: Verify Installation

Check that PyTorch is installed correctly:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.8.0 (or similar)
CUDA available: True (if you have an NVIDIA GPU) or False (if you don't)
```

---

## Understanding the Feature Extractor

The feature extractor is a pre-trained neural network that the script uses to analyze images. Don't worry - **you don't need to download this manually!** The script handles it automatically.

### What Happens Automatically:

When you run the script for the first time, PyTorch will:

1. **Download the pre-trained model** from the internet
   - The script uses EfficientNet-B4 by default
   - This is about 70-80 MB in size
   - It gets downloaded to your home directory in a folder called `.cache/torch/hub/checkpoints/`

2. **Save it for future use**
   - You only download it once
   - Future runs will use the cached version
   - This makes subsequent runs much faster

### Where the Model is Stored:

**On Linux/macOS:**
```
~/.cache/torch/hub/checkpoints/efficientnet_b4-*.pth
```

**On Windows:**
```
C:\Users\YourUsername\.cache\torch\hub\checkpoints\efficientnet_b4-*.pth
```

### First Run vs. Subsequent Runs:

- **First run:** Takes longer because the model needs to be downloaded
- **Subsequent runs:** Much faster since the model is already cached

You don't need to do anything special - just make sure you have an internet connection the first time you run the script!

---

## Dataset Preparation

If you want to use the **dataset mode** (evaluating AIC4-Evaluation dataset), follow these steps:

### Step 1: Download the Dataset

Download the evaluation dataset from:
```
https://drive.google.com/drive/folders/1TlmLVFTIv7bobxInKnBrd8Nmu4T0AalZ?usp=sharing
```

This will be a aic4-evaluation-png.zip file containing reference and distorted images.

### Step 2: Extract the Dataset

Extract the ZIP file to a location on your computer. For example:
```bash
unzip aic4-evaluation-png.zip -d /path/to/your/datasets/
```

---

## Running the Script

The script has two modes: **single** (for comparing one pair of images) and **dataset** (for comparing many images).

### Mode 1: Single Image Pair Evaluation

Use this mode when you want to quickly check the quality of one distorted image compared to its reference.

**Basic syntax:**
```bash
python aic4-eval.py --mode single --ref-img /path/to/reference.png --dist-img /path/to/distorted.png
```

**Example:**
```bash
python aic4-eval.py --mode single --ref-img ./images/original.png --dist-img ./images/compressed.png
```

**What happens:**
1. The script loads both images
2. Processes them through the quality evaluation model
3. Prints the quality score to the terminal

**Example output:**
```
Quality Score: 0.847523
Reference: ./images/original.png
Distorted: ./images/compressed.png
```

**Interpreting the score:**
- **0.9 - 1.0:** Excellent quality, barely any difference
- **0.8 - 0.9:** Good quality, small differences
- **0.7 - 0.8:** Moderate quality, noticeable differences
- **Below 0.7:** Poor quality, significant differences

### Mode 2: Full Dataset Evaluation

Use this mode when you have many images to evaluate and want results saved to a file.

**Basic syntax:**
```bash
python aic4-eval.py --mode dataset --root-dir /path/to/dataset-folder --output results.csv
```

**Example:**
```bash
python aic4-eval.py --mode dataset --root-dir ./dataset-folder --output results.csv
```

**What happens:**
1. The script finds all reference images in the `source/` folder
2. For each reference image, it finds all corresponding distorted images
3. Calculates quality score for each pair
4. Saves results to a CSV file
5. Shows a progress bar so you can track completion

**Example output during processing:**
```
Evaluating images: 100%|████████████████| 5600/5600 [05:23<00:00, 4.64it/s]
Results saved to results.csv
```

**The output CSV file contains:**
```csv
ref_img_name,dis_img_name,quality_score,jnd_mapped
src_001.png,src_001.png-01-01.png,0.9993844032287598,0.007380344183170351
src_001.png,src_001.png-01-02.png,0.9991859793663024,0.009756329974884181
src_001.png,src_001.png-01-03.png,0.9991020560264589,0.010760827245163362
src_001.png,src_001.png-01-04.png,0.9989848732948304,0.012162990596872536
src_001.png,src_001.png-01-05.png,0.9986413717269896,0.016270358697388243
...
```

### Adjusting Number of Workers (Optional)

By default, the script uses 2 worker processes to load images. You can adjust this based on your computer's CPU:

**For faster computers (4+ CPU cores):**
```bash
python aic4-eval.py --mode dataset --root-dir ./dataset-folder --num-workers 4
```

**For slower computers (2 CPU cores):**
```bash
python aic4-eval.py --mode dataset --root-dir ./dataset-folder --num-workers 1
```

**Note:** More workers = faster image loading, but uses more RAM. Start with 2 and increase if your computer can handle it.

---

## Understanding the Output

### Single Mode Output

When you run single mode, you get a simple text output:

```
Quality Score: 0.847523
Reference: ./images/original.png
Distorted: ./images/compressed.png
```

- **Quality Score:** The quality metric (0.0 to 1.0)
- **Reference:** Path to the original image you provided
- **Distorted:** Path to the distorted image you provided

### Dataset Mode Output

When you run dataset mode, you get a CSV file that looks like this:

```csv
ref_img_name,dis_img_name,quality_score,jnd_mapped
src_001.png,src_001.png-01-01.png,0.9993844032287598,0.007380344183170351
src_001.png,src_001.png-01-02.png,0.9991859793663024,0.009756329974884181
src_001.png,src_001.png-01-03.png,0.9991020560264589,0.010760827245163362
src_001.png,src_001.png-01-04.png,0.9989848732948304,0.012162990596872536
src_001.png,src_001.png-01-05.png,0.9986413717269896,0.016270358697388243
...
```

**Columns explained:**
- **ref_img_name:** The name of the reference (original) image
- **dis_img_name:** The name of the distorted image being compared
- **quality_score:** The quality score for this pair (0-1)
- **jnd_mapped**: quality score mapped to JND.
  We use the following mapping function with `b = 0.99` and `a = 15.0`

``` python
a * max(0, b - x)
```

---

## Advanced Options

### Command-Line Arguments Reference

Here's a complete list of all options you can use:

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--mode` | Evaluation mode: `single` or `dataset` | `single` | No |
| `--ref-img` | Path to reference image (single mode only) | None | Yes (for single mode) |
| `--dist-img` | Path to distorted image (single mode only) | None | Yes (for single mode) |
| `--root-dir` | Root directory of dataset (dataset mode only) | (none - must specify) | Yes (for dataset mode) |
| `--output` | Output CSV file path (dataset mode only) | `results.csv` | No |
| `--num-workers` | Number of worker processes for data loading | `2` | No |
| `--percent-features` | Percentage of feature maps to keep (0.0 to 1.0) | `0.7` | No |

### About --percent-features (Advanced Reference)

The `--percent-features` option controls an internal parameter of the quality evaluation algorithm. The default value of 0.7 (70%) has been carefully chosen and works well for most image quality evaluation tasks.

**Default behavior:**
```bash
python aic4-eval.py --mode single --ref-img ref.png --dist-img dist.png
# Uses --percent-features 0.7 automatically
```

This parameter is included for completeness and research purposes, but most users should not modify it. The default value provides a good balance between accuracy and computational efficiency.

---

## Troubleshooting

### Problem: "command not found: python"

**Solution:** Try using `python3` instead:
```bash
python3 aic4-eval.py --mode single --ref-img ref.png --dist-img dist.png
```

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Cause:** You either didn't activate the virtual environment or didn't install the requirements.

**Solution:**
```bash
# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt
```

### Getting Help

If you encounter an error not listed here:

1. **Read the error message carefully** - it usually tells you what's wrong
2. **Check your Python version**: `python --version` (should be 3.10+)
3. **Verify all packages are installed**: `pip list | grep torch`
4. **Try the single mode first** - it's simpler and helps isolate issues
5. **Check file permissions** - make sure you can read the input files and write to the output location

---

## Quick Start Checklist

Use this checklist to make sure you've completed all setup steps:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created (`python3 -m venv venv`)
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Requirements installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted (for dataset mode)
- [ ] Ran test with single mode first
- [ ] Internet connection available (for first run model download)

Once all items are checked, you're ready to use the tool!

---

## Example Workflows

### Workflow 1: Quick Test with Single Image

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Test with one pair
python aic4-eval.py --mode single \
    --ref-img ./test_images/original.png \
    --dist-img ./test_images/compressed.png

# Output: IDFIQA Score: 0.847523
```

### Workflow 2: Evaluate Full Dataset

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run full dataset evaluation
python aic4-eval.py --mode dataset \
    --root-dir ./dataset-folder \
    --output results.csv \
    --num-workers 4

# 3. Check results
head -n 10 results.csv

# 4. Deactivate when done
deactivate
```
