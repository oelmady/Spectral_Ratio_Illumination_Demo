# Parameter Tuning Guide

## Overview
This guide helps you systematically tune parameters for your SR-constrained Retinex vs baseline comparison.

## Key Parameters

### 1. `--iterations` (Retinex iterations)
**What it does:** Controls how many times the algorithm refines the illumination estimate.

**Range to test:** 1, 3, 5, 10, 20
- **Low (1-3):** Fast, preserves more detail, may under-correct shadows
- **High (10-20):** Stronger correction, smoother illumination, may over-blur

**How to tune:**
```bash
# Test different iteration counts
python scripts/run_batch.py --use-model --retinex --baseline-retinex --iterations 3
python scripts/run_batch.py --use-model --retinex --baseline-retinex --iterations 10
python scripts/run_batch.py --use-model --retinex --baseline-retinex --iterations 20

# Compare outputs in results/ folder
```

**What to look for:**
- Too few iterations: shadows still too dark
- Too many: over-smoothed, unnatural lighting
- Sweet spot: 5-10 for most images

---

### 2. `--sigma` (Gaussian blur size)
**What it does:** Controls spatial scale of illumination estimation (how local vs global).

**Range to test:** 5, 10, 15, 25, 50
- **Small (5-10):** Preserves local detail, sensitive to texture
- **Large (25-50):** Smoother global illumination, less texture-dependent

**How to tune:**
```bash
# Test different blur scales
python scripts/run_batch.py --use-model --retinex --sigma 10
python scripts/run_batch.py --use-model --retinex --sigma 25

# Compare shadow regions - smaller sigma preserves texture
```

**What to look for:**
- Too small: texture leaks into illumination estimate
- Too large: misses local shadow boundaries
- Sweet spot: 15-25 for natural images

---

### 3. `--distance` (SR color shift amount)
**What it does:** How much to shift color along spectral ratio direction (for `--sr-correct`).

**Range to test:** 0.3, 0.5, 1.0, 1.5, 2.0
- **Small (0.3-0.5):** Subtle color adjustment
- **Large (1.5-2.0):** Strong physics-based color change

**How to tune:**
```bash
# Test different shift amounts
python scripts/run_batch.py --use-model --sr-correct --distance 0.5
python scripts/run_batch.py --use-model --sr-correct --distance 1.5

# Look at shadow regions - is color shift natural?
```

**What to look for:**
- Too small: no visible improvement over baseline
- Too large: oversaturated, unnatural colors
- Sweet spot: 0.5-1.0 for realistic correction

---

## Systematic Tuning Workflow

### Step 1: Pick Representative Test Images
Choose 3-5 images with:
- Strong shadows (to test illumination estimation)
- Color variation (to test SR constraint effectiveness)
- Different scene types (indoor/outdoor, etc.)

### Step 2: Grid Search (Coarse)
Test all combinations on one image:
```bash
# Generate all combinations
for iter in 3 5 10; do
  for sigma in 10 15 25; do
    python scripts/run_batch.py --use-model --retinex --baseline-retinex \
      --iterations $iter --sigma $sigma --image test_image_stem
  done
done
```

### Step 3: Visual Comparison
Open `results/` and compare side-by-side:
- `*_baseline_retinex_vis.png` (your baseline)
- `*_sr_retinex_vis.png` (your method)
- Original `*_image_8bit.png`

**Look for:**
- Shadow regions: Is your method brighter/more natural?
- Color accuracy: Does SR-constrained version have better/worse color in shadows?
- Artifacts: Any halos, over-smoothing, or color bleeding?

### Step 4: Quantitative Metrics (If you have ground truth)
```python
# Add to your evaluation script
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Compare against ground truth reflectance
mse = np.mean((pred - gt)**2)
psnr_val = psnr(gt, pred, data_range=65535)
ssim_val = ssim(gt, pred, data_range=65535, channel_axis=2)
```

### Step 5: Fine-Tune Best Candidates
Once you find good ranges (e.g., iterations=5-7, sigma=15-20), test finer steps:
```bash
python scripts/run_batch.py --use-model --retinex --iterations 5 --sigma 15
python scripts/run_batch.py --use-model --retinex --iterations 6 --sigma 17
python scripts/run_batch.py --use-model --retinex --iterations 7 --sigma 20
```

---

## Full Comparison Commands

### Baseline vs Your Method (Side-by-side)
```bash
# Run both on all images with your best parameters
python scripts/run_batch.py --use-model --retinex --baseline-retinex \
  --iterations 5 --sigma 15

# Results will have both:
# results/*_baseline_retinex.tiff  (standard Retinex)
# results/*_sr_retinex.tiff        (your SR-constrained Retinex)
```

### With Color Correction
```bash
# Add SR color correction on top of Retinex
python scripts/run_batch.py --use-model --retinex --sr-correct \
  --iterations 5 --sigma 15 --distance 0.7

# Results will have:
# results/*_sr_retinex.tiff     (illumination-corrected)
# results/*_sr_shifted.tiff     (color-corrected)
```

---

## What Good Results Look Like

### For Your Proposal Deliverables:

**1. Spectral-ratio constrained illumination (your Retinex)**
- Shadows are brightened without over-smoothing
- Colors in shadow regions shift according to physics (not preserved)
- Fewer artifacts than baseline at shadow boundaries

**2. Baseline Retinex (for comparison)**
- Shadows brightened but colors preserved (wrong physics)
- May have color inconsistencies at shadow edges

**3. Quantitative improvements to report:**
- Lower MSE on MIT dataset reflectance ground truth
- Better SSIM scores
- Lower angular error in shadow regions

**4. Qualitative improvements to show:**
- Side-by-side: shadow region zoom-ins
- Highlight: "Baseline preserves blue tint in shadow, ours shifts to yellow (correct physics)"

---

## Quick Reference

```bash
# Best starting point (recommended defaults)
python scripts/run_batch.py --use-model --retinex --baseline-retinex \
  --iterations 5 --sigma 15 --distance 1.0

# Conservative (safer, less aggressive)
python scripts/run_batch.py --use-model --retinex --iterations 3 --sigma 10

# Aggressive (stronger correction)
python scripts/run_batch.py --use-model --retinex --iterations 10 --sigma 25

# Just color correction (no full Retinex)
python scripts/run_batch.py --use-model --sr-correct --distance 0.5
```

---

## Troubleshooting

**Problem: Results look over-smoothed**
- Reduce `--iterations` or `--sigma`

**Problem: Shadows still too dark**
- Increase `--iterations`

**Problem: Texture bleeding into illumination**
- Increase `--sigma`

**Problem: Colors look unnatural**
- For SR-Retinex: This might actually be correct! Compare to baseline.
- For SR-correct: Reduce `--distance`

**Problem: No visible difference between baseline and SR-Retinex**
- Check your SR maps are being loaded correctly
- Try increasing `--iterations` to make differences more pronounced
- Look specifically at shadow-to-light transition regions
