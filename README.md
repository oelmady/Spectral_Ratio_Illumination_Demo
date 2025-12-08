# Spectral Ratio Illumination Decomposition - Submission Instructions

**Student:** Omar Elmady  
**Course:** CS 7180 - Advanced Perception  
**Date:** December 11, 2025

---

## Project Overview

This project implements and extends spectral-ratio constrained illumination decomposition algorithms, including:

- **Baseline Retinex**: Standard chromaticity-preserving Retinex algorithm
- **SR-Constrained Retinex**: Novel spectral-ratio constrained version that projects illumination updates onto per-pixel spectral-ratio directions
- **SR-Based Color Correction**: Color correction using spectral-ratio direction shifting in log-space
- **Neural ISD Prediction**: ResNet50-UNet model for predicting illuminant spectral direction maps

---

## How to Run This Project

### ✅ Recommended Method: Google Colab (No Setup Required)

This is the **easiest and fastest** way to run the project with GPU acceleration.

**Steps:**

1. **Open the Colab Notebook**
   - File: `run_on_colab.ipynb`
   - Upload to Google Colab or open directly if already there

2. **Enable GPU Runtime**
   - In Colab: Runtime → Change runtime type → Hardware accelerator → **GPU** → Save

3. **Run All Cells**
   - Click: Runtime → Run all
   - The notebook will automatically:
     - Download the model from Google Drive
     - Clone the GitHub repository
     - Install all dependencies
     - Run experiments
     - Generate and download results

4. **Download Results**
   - Results automatically download as `results.tar.gz` at the end
   - Extract on your computer: `tar -xzf results.tar.gz`

**Model File:** The pre-trained model (~528MB) is pre-configured in the notebook and will download automatically from:
```
https://drive.google.com/file/d/1h2fVtLQJpgLl4_C3MLA_VDuqlJTcAqf6/view?usp=share_link
```

**Expected Runtime:** ~10-15 minutes for standard experiments (GPU) or ~30-45 minutes (CPU)

---

### 🔧 Alternative Method: Local/Remote Execution via GitHub

If you prefer to run locally or on a remote server:

**Repository:** https://github.com/oelmady/Spectral_Ratio_Illumination_Demo

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.0+
- NumPy, Matplotlib

**Setup:**
```bash
# Clone repository
git clone https://github.com/oelmady/Spectral_Ratio_Illumination_Demo.git
cd Spectral_Ratio_Illumination_Demo

# Install dependencies
pip install -r requirements.txt

# Download model file from Google Drive link above
# Place in: model/UNET_run_x10_01_last_model.pth

# Run validation
python preflight_check.py

# Run experiments
python scripts/run_batch.py --use-model --retinex --baseline-retinex --sr-correct
```

**Documentation Available:**
- `README.md` - Project overview and usage
- `TUNING_GUIDE.md` - Parameter tuning methodology
- `README_EXPERIMENTS.md` - Complete experiment workflow
- `REMOTE_SETUP.md` - Remote server setup instructions

---

## Project Structure

```
Spectral_Ratio_Illumination_Demo/
├── run_on_colab.ipynb          # Google Colab notebook (RECOMMENDED)
├── run.py                       # Interactive processing script
├── scripts/
│   └── run_batch.py            # Batch processing for all images
├── algorithms/
│   └── retinex.py              # Retinex implementations
├── model/
│   ├── unet_models2.py         # ResNet50-UNet architecture
│   └── UNET_run_x10_01_last_model.pth  # Pre-trained model (download required)
├── data/
│   ├── images/                 # Input 16-bit TIFF images
│   └── sr_maps/                # Ground-truth spectral-ratio maps
├── results/                    # Output directory (generated)
├── preflight_check.py          # Validation script
└── Documentation files
```

---

## What the Code Does

### Core Algorithms Implemented

1. **Baseline Retinex** (`algorithms/retinex.py::baseline_retinex()`)
   - Standard Retinex with Gaussian blur-based illumination estimation
   - Chromaticity-preserving with McCann-Sobel color restoration

2. **SR-Constrained Retinex** (`algorithms/retinex.py::spectral_ratio_retinex()`)
   - Novel contribution: Projects illumination updates onto spectral-ratio directions
   - Uses per-pixel SR maps from neural network or ground-truth annotations

3. **SR Color Correction** (`algorithms/retinex.py::apply_spectral_ratio_color_correction()`)
   - Shifts pixels in log-space along spectral-ratio vectors
   - Tunable distance parameter for correction strength

### Neural Network

- **Architecture:** ResNet50 encoder + UNet decoder with skip connections
- **Input:** 16-bit linear RGB images (log-transformed)
- **Output:** 2-channel ISD maps (normalized spectral-ratio directions)
- **Training:** Pre-trained on spectral illumination dataset

### Batch Processing

- Processes all images in `data/images/` directory
- Generates multiple outputs per image:
  - ISD maps (TIFF and PNG visualization)
  - Retinex outputs (baseline and SR-constrained)
  - Color-corrected outputs
  - 8-bit reference images

---

## Expected Outputs

After running experiments, the `results/` directory will contain:

```
results/
├── <image_name>_isd_map.tif           # Neural ISD prediction (2-channel float)
├── <image_name>_isd_map.png           # ISD visualization (RGB)
├── <image_name>_retinex.tif           # SR-constrained Retinex output
├── <image_name>_retinex.png           # SR-constrained visualization
├── <image_name>_baseline_retinex.tif  # Standard Retinex output
├── <image_name>_baseline_retinex.png  # Baseline visualization
├── <image_name>_sr_corrected.tif      # Color-corrected output
├── <image_name>_sr_corrected.png      # Color correction visualization
└── <image_name>_8bit.png              # Input as 8-bit reference
```

---

## Validation

Run the preflight check to verify setup:

```python
python preflight_check.py
```

This checks:
- ✅ All Python dependencies installed
- ✅ Project structure intact
- ✅ Algorithm implementations present
- ✅ Model file exists and correct size
- ✅ Sample data available

---

## Parameter Tuning

The Colab notebook includes an optional automated parameter tuning section that tests:

- **Iterations:** 3, 5, 10 (Retinex iterations)
- **Sigma:** 10, 15, 20, 25 (Gaussian blur amount)
- **Distance:** 0.5, 1.0, 1.5 (color correction strength)

This generates 36 parameter combinations and saves results separately for comparison.

---

## Troubleshooting

### Model download issues
- Ensure Google Drive permissions are set to "Anyone with the link can view"
- Check file ID is correct: `1h2fVtLQJpgLl4_C3MLA_VDuqlJTcAqf6`
- If quota exceeded, wait or request new link

### No results generated
- Verify `data/images/` contains `.tif` or `.tiff` files
- Check for error messages in console/notebook output
- Run `preflight_check.py` to diagnose issues

### Memory errors
- Switch to CPU runtime in Colab (Runtime → Change runtime type → None)
- Process fewer images at once

### Import errors
- Re-run dependency installation
- Verify Python version is 3.8+

---

## Academic Integrity Statement

This project represents my original work for CS 7180. The base spectral-ratio illumination decomposition framework was provided as starter code. My contributions include:

1. Implementation of baseline Retinex algorithm
2. Implementation of spectral-ratio constrained Retinex
3. Implementation of SR-based color correction
4. Batch processing pipeline with parameter tuning
5. Google Colab notebook for reproducible execution
6. Documentation and validation tools

All code is available in the GitHub repository with clear commit history showing development progression.
