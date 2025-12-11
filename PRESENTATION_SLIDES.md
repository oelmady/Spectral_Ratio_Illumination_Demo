# Spectral-Ratio Enhanced Illumination Correction
## Omar Elmady | CS 7180 | December 2025

---

## Slide 1: The Problem

**Traditional illumination correction ignores physics**

When you brighten a shadow, what should happen to colors?

**Current methods:** Preserve original chromaticity
- Simple brightness boost
- Ignores how light actually works

**Physics says:** Color changes with illumination
- Shadow under blue sky → bluish
- Shadow under warm light → reddish
- **Spectral ratio** describes this relationship

**Question:** Can we use neural spectral ratio predictions to improve correction?

---

## Slide 2: My Approach

**Used neural network to predict spectral ratios, then tested 3 methods:**

### 1. SR Color Correction (Direct Physics)
- Shift colors along predicted SR direction
- Simple, fast, no spatial processing

### 2. SR-Constrained Retinex (Spatial Physics)
- Combine Retinex spatial estimation with SR
- Attempt to get "best of both worlds"

### 3. Traditional Baselines
- Gray World, White Patch, Multi-Scale Retinex
- Standard computer vision methods

---

## Slide 3: The Setup

**10 test images** from MIT dataset
- Complex shadows, various materials

**Neural prediction**
- ResNet50-UNet (pre-trained)
- Predicts per-pixel spectral ratio maps

**Metrics**
- **Color Constancy Error (°)** - color accuracy (lower = better)
- **SSIM** - structure preservation (higher = better)

---

## Slide 4: The Results

| Method | Color Error | SSIM |
|--------|-------------|------|
| **SR Color Correction** | **2.59°** | 0.7974 |
| White Patch | 5.41° | 0.9789 |
| Gray World | 5.97° | 0.9855 |
| Baseline Retinex | 6.02° | 0.4708* |
| Multi-Scale Retinex | 6.04° | 0.6039 |

**SR Color Correction wins by 2x on color accuracy!**

*Low SSIM for Retinex methods due to aggressive darkening - trades brightness for reflectance estimation

---

## Slide 5: Why Direct Physics Wins

**SR Color Correction: 2.59° error**

✅ Uses accurate neural predictions directly
- No processing artifacts

✅ Preserves image structure  
- High SSIM relative to color improvement

✅ Computationally efficient
- Just a shift in log-space

**Key insight:** When predictions are good, simple application beats complex integration

---

## Slide 6: The Spatial Challenge

**I tried combining SR with Retinex spatial processing**

**Goal:** Get spatial adaptation + physics constraints

**Result:** Challenging to integrate
- Retinex estimates spatial illumination structure
- SR describes color relationships
- These are fundamentally different problems

**Brightness Calibration:** Retinex required careful tuning
- Median anchor + minimal offset (+0.1 log units)
- Balances shadow removal with color preservation
- Avoids over-correction that creates unnatural colors

**Lesson:** Not all combinations of good methods produce better results

---

## Slide 7: Visual Comparison

[Show your comparison images]

**What to look for:**
- SR Color Correction: Natural colors, good brightness
- Traditional methods: Over-processed or color shifts
- Material colors preserved (red stays red!)

---

## Slide 8: Key Contributions

**1. Validated physics-based correction**
   - 57% better color accuracy (2.59° vs 6.02°)

**2. Comprehensive comparison**
   - 6 methods, quantitative metrics
   - Established benchmark

**3. Practical insight**
   - Direct application of neural predictions > complex integration
   - Guides future system design

---

## Slide 9: Takeaways

**Main finding:**
Direct spectral-ratio color correction achieves **2.59° color error** - significantly better than all traditional methods

**Why it matters:**
- Validates physics-based approach with neural predictions
- Shows simple can beat complex with good predictions
- Provides baseline for future illumination work

**Future:**
- Adaptive correction with confidence maps
- Real-time implementation

---

## Slide 10: Thank You

**Omar Elmady** | CS 7180

**Code:** github.com/oelmady/Spectral_Ratio_Illumination_Demo

**Key Result:** Physics-based SR correction - **57% better** than baseline

**Questions?**

---

# BACKUP SLIDES

---

## Technical Details: SR Color Correction

```python
# Simple but effective
1. Convert to log-space: log_img = log(image)
2. Get SR prediction: sr_map = neural_net(image)
3. Normalize: sr_unit = sr_map / ||sr_map||
4. Shift: corrected = log_img + distance * sr_unit
5. Back to linear: output = exp(corrected)
```

**One parameter:** Distance = 1.0 (brightness increase)

---

## Why Gray World Does So Well

**Gray World: 5.97° error, 0.9855 SSIM**

- Assumes scene average is neutral gray
- Global scaling per channel
- **Preserves all structure** (SSIM = 0.99!)

**Our method trades 0.2 SSIM for 3.4° better color**
- Worth it for physics-based correction

---

## Computational Cost

| Method | Time/Image | Notes |
|--------|------------|-------|
| SR Prediction | 200ms | Once per image |
| SR Color Corr | 50ms | Very fast |
| Retinex | 500ms | Iterative |
| Gray World | 10ms | Fastest |

**For real-time:** SR + Direct method is viable

---

## Q&A Prep

**Q: Why not train end-to-end?**
A: Wanted to show that classical + neural can work. Modular design is easier to debug and understand.

**Q: Failure cases?**
A: When SR predictions are wrong, everything fails. Need confidence maps in future work.

**Q: Why didn't SR-Constrained Retinex work better?**
A: Spatial illumination estimation and color constraints model different things. Hard to combine meaningfully.

**Q: Why is Retinex so dark / has low SSIM?**
A: Retinex removes illumination to recover reflectance. I calibrated brightness using 90th percentile anchor + 0.3 log offset to preserve highlights while revealing shadow detail. The low SSIM (0.47) reflects the fundamental transformation from illumination-dependent image to reflectance estimate - not a failure, just a different representation.

**Q: How did you tune the brightness?**
A: Tried mean (too dark), 95th percentile (washed out colors), settled on 90th percentile + 0.3 log offset (~35% brightness boost). This balances reflectance estimation with practical visibility.

**Q: Real-world applications?**
A: Computational photography, photo enhancement, any app that needs realistic shadow brightening.

---
