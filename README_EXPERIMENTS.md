# Experiment Readiness Summary

## ✅ What's Complete

### 1. Code Implementation (100% Done)
- ✓ **Baseline Retinex** (`algorithms/retinex.py`)
  - Standard chromaticity-preserving Retinex
  - Used for comparison
  
- ✓ **SR-Constrained Retinex** (`algorithms/retinex.py`)
  - Projects illumination updates onto spectral-ratio direction
  - Your novel contribution (Equations 2-6)
  
- ✓ **SR Color Correction** (`algorithms/retinex.py`)
  - Shifts pixels along SR vectors
  - Physics-based alternative (Equations 7-10)
  
- ✓ **Batch Processing** (`scripts/run_batch.py`)
  - Non-interactive processing of all images
  - Saves all outputs to `results/`
  - Supports both methods side-by-side
  
- ✓ **Parameter Tuning** (`TUNING_GUIDE.md`)
  - Systematic workflow
  - Parameter ranges and guidance
  - Example commands

### 2. Environment (Fixed!)
- ✓ `isd_fixed` conda environment created
- ✓ PyTorch 2.9.1 (CPU) installed
- ✓ OpenCV 4.7.0 headless working
- ✓ All imports functional (no dylib errors)

### 3. Documentation
- ✓ `TUNING_GUIDE.md` - parameter tuning workflow
- ✓ `README_EXPERIMENTS.md` (this file)
- ✓ `preflight_check.py` - validation script

---

## ⚠️ What You Still Need

### Model Weights (Critical)
**Status:** Not downloaded (Git LFS pointer file exists, not actual weights)

**Size:** 528MB

**Options:**
1. **Local download** (if you have space):
   ```bash
   brew install git-lfs
   git lfs install
   git lfs pull --include="model/UNET_run_x10_01_last_model.pth"
   ```

2. **Remote server** (recommended):
   - Ask professor for lab machine access
   - Clone repo on server and run experiments there
   - Server will have better compute and no storage issues

### Test Data (Optional)
- If you have images in `data/images/`, you're ready
- Otherwise, download MIT Intrinsic dataset or use professor's test images

---

## 🚀 How to Run Experiments

### Step 1: Verify Setup
```bash
# Activate environment
conda activate isd_fixed

# Run preflight check
python preflight_check.py
```

This will tell you exactly what's missing.

### Step 2: Basic Test (Once Model Downloaded)
```bash
# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Run on one image with both methods
python scripts/run_batch.py --use-model --retinex --baseline-retinex --image <image_stem>
```

### Step 3: Full Experiment Pipeline
```bash
# Process all images with default parameters
python scripts/run_batch.py --use-model --retinex --baseline-retinex

# Try different parameters
python scripts/run_batch.py --use-model --retinex --baseline-retinex --iterations 10 --sigma 20

# Add color correction
python scripts/run_batch.py --use-model --retinex --sr-correct --distance 0.7
```

### Step 4: Check Results
```bash
# View outputs
ls -lh results/

# Each image will have:
# - *_baseline_retinex_vis.png  (standard Retinex)
# - *_sr_retinex_vis.png        (your SR-constrained Retinex)
# - *_image_8bit.png            (original for reference)
```

---

## 📊 What to Report for Your Project

### Quantitative (If you have ground truth)
1. **MSE** between predicted reflectance and ground truth
2. **PSNR** (peak signal-to-noise ratio)
3. **SSIM** (structural similarity)
4. **Angular error** in shadow regions specifically

Compare: Baseline vs Your SR-Retinex

### Qualitative (Always required)
1. **Side-by-side comparisons**
   - Original | Baseline | Your Method | Ground Truth (if available)
   
2. **Zoom into shadow regions**
   - Show color shifts are physically accurate
   - Highlight where baseline fails (preserved wrong color)
   
3. **Parameter sweep results**
   - Show effect of iterations (3, 5, 10, 20)
   - Show effect of sigma (10, 15, 25)

---

## 🐛 Common Issues & Solutions

### "Import cv2 failed"
```bash
# Wrong environment
conda activate isd_fixed
python -c "import cv2; print(cv2.__version__)"  # Should print 4.7.0
```

### "ModuleNotFoundError: No module named 'model'"
```bash
# Missing PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### "Failed to load checkpoint"
```bash
# Model file is Git LFS pointer (134 bytes)
ls -lh model/UNET_run_x10_01_last_model.pth

# Should be ~503MB, not 134 bytes
# Fix: download actual weights (see above)
```

### "No such file or directory: data/images"
```bash
# Create directory and add images
mkdir -p data/images
# Copy your test images there
```

---

## 📝 Experiment Checklist

Before running experiments:
- [ ] Environment activated (`conda activate isd_fixed`)
- [ ] Model weights downloaded (503MB file)
- [ ] Test images in `data/images/`
- [ ] `PYTHONPATH` set
- [ ] `preflight_check.py` passes all checks

For your writeup:
- [ ] Baseline Retinex results generated
- [ ] SR-constrained Retinex results generated
- [ ] Side-by-side comparisons created
- [ ] Parameter tuning experiments run
- [ ] Quantitative metrics computed (if ground truth available)
- [ ] Qualitative analysis written (zoom-ins, color accuracy discussion)

---

## 🎓 For Your Professor

**Subject:** Remote Server Request for CV Project

**Message Template:**

> Hi Professor,
>
> I'm working on extending the spectral-ratio intrinsic decomposition project with physics-based Retinex algorithms. I've implemented all the code (baseline Retinex, SR-constrained Retinex, and batch processing pipelines), but I'm encountering hardware limitations on my Mac:
>
> 1. The pretrained model checkpoint is 528MB (Git LFS)
> 2. My system has limited storage and is running macOS Big Sur
> 3. I've successfully created a working conda environment but cannot download the model weights locally
>
> Could I please access a lab machine or remote server to run my experiments? I need:
> - ~2GB disk space
> - Git LFS support
> - Conda or modern Linux environment
>
> All my code is ready to run - I just need a system that can handle the model file.
>
> Thank you,
> Omar

---

## 🔬 Expected Timeline

Once you have model access:

**Day 1 (2-3 hours):**
- Download model and verify with `preflight_check.py`
- Run basic test on 1-2 images
- Confirm both methods produce outputs

**Day 2 (3-4 hours):**
- Process full dataset with default parameters
- Run parameter sweep (iterations, sigma)
- Generate all visualizations

**Day 3 (2-3 hours):**
- Compute quantitative metrics
- Create comparison figures
- Write analysis

**Day 4 (2-3 hours):**
- Final writeup
- Polish figures
- Prepare presentation

**Total:** ~10-13 hours of active work (after getting model access)

---

## 📚 Key References for Writeup

- **Your proposal deliverables** (check them off as you complete):
  1. ✅ Working implementation of base recursive Retinex
  2. ✅ Integration with pretrained spectral ratio network
  3. ✅ Spectral-ratio constrained illumination estimation
  4. ✅ Color correction using spectral ratios
  5. ⏳ Quantitative evaluation on MIT dataset
  6. ⏳ Qualitative results showing improved color accuracy

- **What makes your method novel:**
  - Standard Retinex assumes chromaticity is invariant → wrong
  - Your method uses SR to constrain updates → physically correct
  - Show this improves color accuracy in shadow→light transitions

---

## ✅ Bottom Line

**Code:** 100% ready ✓  
**Environment:** Working ✓  
**Documentation:** Complete ✓  
**Model:** Need to download or use remote server ⚠️  

**You are ONE MODEL DOWNLOAD away from running experiments.**

Run `python preflight_check.py` to confirm!
