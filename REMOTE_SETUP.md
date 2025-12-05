# Remote Server Setup Guide

Step-by-step instructions to deploy and run your project on the remote server.

## Phase 1: Initial Connection & Project Setup

### Step 1.1: SSH into the Remote Server
```bash
ssh username@remote.server.address
# Or if your professor provides a specific command, use that
```

**What to expect:**
- You'll be prompted for a password (ask professor if needed)
- You should see a terminal prompt like `username@server:~$`

### Step 1.2: Verify Python is Available
```bash
python3 --version
# Expected output: Python 3.8+ (ideally 3.10 or higher)
```

If Python isn't available, ask your professor to install it or use their existing Python environment.

### Step 1.3: Clone Your Project Repository
```bash
git clone https://github.com/oelmady/Spectral_Ratio_Illumination_Demo.git
cd Spectral_Ratio_Illumination_Demo
```

**What this does:**
- Downloads all your code files (except the 528MB model, which git-lfs will handle next)
- Creates a `Spectral_Ratio_Illumination_Demo` directory
- Navigates into the project folder

### Step 1.4: Download the Model Using Git-LFS
```bash
# Check if git-lfs is already installed
git lfs version

# If installed, pull the model (this may take 1-2 minutes)
git lfs pull --include="model/UNET_run_x10_01_last_model.pth"

# Verify the model was downloaded (should be ~528MB, not 134 bytes)
ls -lh model/UNET_run_x10_01_last_model.pth
```

**Expected output for model file:**
```
-rw-r--r-- 1 username group 528M Dec 4 12:34 model/UNET_run_x10_01_last_model.pth
```

If git-lfs is NOT installed on the server, ask your professor to install it, or they may have already pulled it for you.

---

## Phase 2: Environment Setup

### Step 2.1: Set Up Python Virtual Environment (Recommended Approach)
This isolates your project dependencies from system Python.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
# You should see (venv) in your terminal prompt

# Upgrade pip to latest version
pip install --upgrade pip
```

**Alternative: Use Conda (If Available)**
If the server has conda, you can use it instead:
```bash
conda create -n spectral python=3.11
conda activate spectral
```

### Step 2.2: Install Project Dependencies
```bash
# Make sure you're in the project directory
cd ~/Spectral_Ratio_Illumination_Demo

# Install all required packages
pip install -r requirements.txt

# This will install: opencv-python, torch, torchvision, numpy, matplotlib
```

**What to expect:**
- Installation may take 2-5 minutes (downloading PyTorch is ~500MB)
- You should see `Successfully installed` messages for each package

### Step 2.3: Verify Installation
```bash
# Test that all imports work
python preflight_check.py
```

**Expected output:**
```
========================================
PREFLIGHT CHECK
========================================
✓ All imports successful
✓ Project structure valid
✓ Algorithm implementations present
✓ Model checkpoint found (528MB)
✓ Data directory exists

STATUS: READY FOR EXPERIMENTS
========================================
```

If any checks fail, refer to the troubleshooting section at the end of this guide.

---

## Phase 3: Running Experiments

### Step 3.1: Prepare Input Data
Your input images should be 16-bit linear RGB TIFF files in `data/images/`:

```bash
# Check what data is already there
ls -la data/images/

# If you need to upload images from your local machine:
# On your LOCAL Mac (in a separate terminal):
# scp /path/to/your/image.tif username@remote.server.address:~/Spectral_Ratio_Illumination_Demo/data/images/
```

### Step 3.2: Run the Batch Processor
This runs the ISD model and all Retinex variations on all images:

```bash
# Make sure venv is activated
source venv/bin/activate  # or: conda activate spectral

# Run the batch processor
python scripts/run_batch.py \
  --use-model \
  --retinex \
  --baseline-retinex \
  --sr-correct \
  --iterations 5 \
  --sigma 15 \
  --distance 1.0
```

**What this does:**
- Processes all images in `data/images/`
- Runs the neural network ISD predictor on each
- Applies baseline Retinex for comparison
- Applies SR-constrained Retinex
- Applies SR-based color correction
- Writes outputs to `results/` (maps as TIFF, visualizations as PNG)

**Expected runtime:**
- Model inference: ~30 seconds per image (CPU)
- Retinex algorithms: ~5-10 seconds per image
- Total for 10 images: ~5-10 minutes

### Step 3.3: Download Results Back to Your Mac
Once experiments complete, retrieve the results:

```bash
# On your REMOTE terminal, create a tar archive of results
tar -czf results.tar.gz results/

# On your LOCAL Mac (in a separate terminal):
scp username@remote.server.address:~/Spectral_Ratio_Illumination_Demo/results.tar.gz ./
tar -xzf results.tar.gz
# Results now in ./results/ locally
```

### Step 3.4: Parameter Tuning (Optional)
For systematic parameter exploration, follow the guide in `TUNING_GUIDE.md`:

```bash
# Example: Test different sigma values
for SIGMA in 10 15 20 25 30; do
  echo "Running with sigma=$SIGMA"
  python scripts/run_batch.py \
    --use-model \
    --retinex \
    --baseline-retinex \
    --iterations 5 \
    --sigma $SIGMA
done
```

---

## Phase 4: Troubleshooting

### Issue: `git lfs pull` returns empty or small file
**Cause:** git-lfs not installed on server
**Solution:** 
```bash
# Ask your professor to run:
sudo apt-get install git-lfs  # Linux
# or
brew install git-lfs  # macOS
# Then retry: git lfs pull --include="model/UNET_run_x10_01_last_model.pth"
```

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Cause:** Dependencies not installed
**Solution:**
```bash
# Make sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate
# Reinstall
pip install -r requirements.txt
```

### Issue: `CUDA out of memory` or extremely slow inference
**Cause:** GPU not available or model trying to use it
**Solution:** Model defaults to CPU (safe for all systems). If slow, try:
```bash
# Reduce batch size if batch processing hangs
python scripts/run_batch.py --use-model --batch 1  # Process one image at a time
```

### Issue: `Permission denied` when writing to `results/`
**Cause:** Directory doesn't exist or wrong permissions
**Solution:**
```bash
mkdir -p results/
chmod 755 results/
```

### Issue: Images in `data/images/` not being processed
**Cause:** Wrong file format or directory structure
**Solution:**
```bash
# Verify directory structure
ls -la data/images/
# Should show .tif or .tiff files

# Check file format
file data/images/sample.tif
# Should show "TIFF image data"
```

---

## Quick Reference: Essential Commands

### Activation (run at start of each session)
```bash
ssh username@remote.server.address
cd ~/Spectral_Ratio_Illumination_Demo
source venv/bin/activate  # or: conda activate spectral
```

### Run Full Experiment
```bash
python scripts/run_batch.py --use-model --retinex --baseline-retinex --sr-correct
```

### Check Progress
```bash
ls -la results/ | head -20
```

### Download Results
```bash
# On remote:
tar -czf results.tar.gz results/

# On your Mac (local terminal):
scp username@remote.server.address:~/Spectral_Ratio_Illumination_Demo/results.tar.gz ./
```

### Deactivate Environment (when done)
```bash
deactivate  # or: conda deactivate
logout  # Exit remote server
```

---

## Contact Professor If:
- git-lfs is not installed and you can't run `git lfs version`
- Python 3.8+ is not available
- You don't have write permissions to the project directory
- Installation step 2.2 fails with permission errors
- Model inference is extremely slow (>5 minutes per image) suggesting GPU issues

---

## Success Indicators
✓ `preflight_check.py` returns all green checks
✓ Model file is ~528MB (not 134 bytes)
✓ `python scripts/run_batch.py --use-model` completes without errors
✓ `results/` directory contains TIFF and PNG files
✓ You can download results back to your Mac with `scp`
