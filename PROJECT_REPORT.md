# Spectral-Ratio Constrained Retinex for Intrinsic Image Decomposition

**Omar Elmady** | CS 7180 - Advanced Perception | December 11, 2025

---

## Abstract

This project implements spectral-ratio constrained Retinex for physically accurate illumination decomposition. Standard Retinex preserves chromaticity, violating physics when illumination color changes. I implemented three methods: (1) baseline Retinex, (2) **SR-constrained Retinex** (projects illumination updates onto per-pixel spectral directions), and (3) SR-based color correction. A pre-trained ResNet50-UNet predicts spectral direction maps. Results show **improved color preservation** and physically plausible decomposition vs. baseline.

---

## 1. Problem & Motivation

**Challenge**: Separate images into reflectance (material) and illumination (lighting) components.

**Standard Retinex Limitation**: Preserves pixel chromaticity when brightening shadows → violates physics. When illumination changes (shadow→sunlight), reflected light color **should** change according to spectral ratios, not stay proportional.

**Real-world need**: Outdoor scenes mix sunlight (warm, 5500K) and skylight (cool, 10000K). Indoor scenes mix artificial + natural light. Accurate decomposition must respect these physical constraints.

**Solution**: Constrain Retinex illumination updates to align with per-pixel **Illuminant Spectral Directions (ISD)**, learned by a neural network.

---

## 2. Method Overview

### Pipeline
```
16-bit RGB Image → Neural ISD Prediction (ResNet50-UNet) 
                 → SR-Constrained Retinex / Baseline / Color Correction
                 → Illumination + Reflectance
```

### Three Algorithms Implemented

#### 2.1 Baseline Retinex (Reference)
```python
# Iteratively refine illumination via spatial filtering
I^(t+1) = GaussianBlur(log(img) - (GaussianBlur(log(img)) - I^t))
R = log(img) - I  # Reflectance
```
**Issue**: Preserves chromaticity → unphysical color changes

#### 2.2 SR-Constrained Retinex (Our Method)
```python
# Gentle illumination compensation with SR guidance
I_mean = mean(I)
I_normalized = I_mean + 0.5 * (I - I_mean)  # 50% compression
corrected = log(img) - (I - I_normalized)   # Partial compensation

# Apply gentle SR color constraint (10% strength)
sr_shift = project_onto_SR(corrected) * 0.1
final = corrected + sr_shift
```
**Benefit**: Physics-guided color correction + gentle illumination normalization → 9% better color accuracy

#### 2.3 SR Color Correction
```python
# Direct shift along spectral vector (fast, simple)
corrected = log(img) + distance * sr_unit
```
**Use case**: Quick color cast removal, white balance

### Neural ISD Prediction
- **Architecture**: ResNet50 encoder + UNet decoder
- **Input**: 16-bit linear RGB (log-transformed)
- **Output**: Per-pixel unit vectors indicating illumination direction
- **Training**: Pre-trained on annotated dataset (528MB model)

---

## 3. Results

### Quantitative Evaluation

Tested on **10 images** from MIT Intrinsic Images dataset with ground-truth reflectance annotations.

**Metrics**:
- **Color Constancy Error (degrees)**: Angular error between estimated and ground-truth reflectance (lower = better)
- **SSIM**: Structural similarity to original image (higher = better for subtle correction)

| Method | Color Error (°) | SSIM | Notes |
|--------|----------------|------|-------|
| **SR Color Correction** | **2.59** | 0.7974 | **Best color accuracy** |
| White Patch | 5.41 | 0.9789 | High structure preservation |
| **SR-Constrained Retinex** | **5.47** | 0.6936 | **9% better than baseline** |
| Gray World | 5.97 | 0.9855 | Simple but effective |
| Baseline Retinex | 6.00 | 0.6913 | Reference method |
| Multi-Scale Retinex | 6.04 | 0.6039 | Multiple scales, similar to baseline |

**Key Findings**:
1. **SR Color Correction dominates**: 2.59° error (57% better than baseline) - direct physics application wins
2. **SR-Constrained Retinex shows improvement**: 5.47° vs 6.00° baseline (9% better color, similar SSIM)
3. **Simple methods competitive**: Gray World/White Patch achieve good results (high SSIM, moderate color error)
4. **Trade-off observed**: Retinex methods sacrifice SSIM (~0.69) for illumination normalization

### Qualitative Findings

**SR-Constrained vs Baseline**:
- ✅ **Better color preservation**: 9% lower color error, materials maintain hues
- ✅ **Natural appearance**: Gentle 50% illumination compensation (vs aggressive normalization)
- ✅ **Physics-guided**: SR constraint provides subtle color correction
- ⚠️ **Modest improvement**: Effect is subtle - simple methods also effective

**Baseline Retinex Issues**:
- ❌ Slightly higher color error (6.00° vs 5.47°)
- ❌ No physics constraint - purely spatial smoothing
- ⚠️ Similar SSIM (~0.69) - both methods transform image significantly

### Parameter Sensitivity

Tested: iterations ∈ {3,5,10}, sigma ∈ {10,15,25}, distance ∈ {0.5,1.0,1.5}

**Key findings**:
- **Sigma (blur)** most critical: 15-20 optimal for natural-looking smoothness
- **Iterations**: 5-7 sufficient (diminishing returns after)
- **Distance**: 0.8-1.2 for color correction (higher = stronger effect)

### Performance

Google Colab (Tesla T4 GPU):
- Neural ISD prediction: **2.1s/image**
- SR-Constrained Retinex: **1.2s/image**
- Baseline Retinex: **0.8s/image**
- Color correction: **0.1s/image**
- **Full pipeline: ~4s/image**

---

## 4. Discussion

### Key Contributions
1. **Physically-constrained Retinex**: Applying SR constraint to Retinex achieves 9% improvement in color accuracy (5.47° vs 6.00°)
2. **Direct physics wins**: SR Color Correction (2.59°) significantly outperforms spatial methods - simple approach best when predictions are accurate
3. **Comprehensive comparison**: Evaluated 6 methods (SR-constrained, baseline, MSR, Gray World, White Patch, SR color correction) with quantitative metrics
4. **Practical pipeline**: Complete system with Google Colab notebook for reproducibility

### Insights
- **Spatial vs. color constraints**: Combining Retinex spatial processing with SR color physics is challenging - they model different aspects (illumination structure vs. color relationships)
- **When simple wins**: Direct SR color correction outperforms complex methods when neural predictions are accurate
- **Illumination normalization trade-off**: Retinex methods achieve lower SSIM (~0.69) because they fundamentally transform the image; simple methods preserve structure (0.98+ SSIM) but higher color error

### Limitations
- **Modest SR-Retinex improvement**: Only 9% better than baseline - suggests spatial and color constraints are somewhat orthogonal
- **ISD quality dependence**: All SR methods require accurate spectral ratio predictions
- **Two-illuminant assumption**: Trained for sun/sky; may not generalize to complex indoor scenes
- **SSIM trade-off**: Retinex methods sacrifice structural similarity for illumination correction

### Future Work
- End-to-end learning (train network to output SR-constrained decomposition directly)
- Multi-illuminant extension (>2 light sources)
- Automatic parameter selection from image analysis
- Real-time GPU implementation for video

---

## 5. Conclusion

This work demonstrates that **direct physics-based color correction outperforms complex spatial methods** when neural predictions are accurate. Key results:

1. **SR Color Correction achieves 2.59° color error** - 57% better than baseline Retinex (6.00°), validating direct application of spectral ratio physics

2. **SR-Constrained Retinex shows 9% improvement** over baseline (5.47° vs 6.00°) - modest but consistent gain from physics constraint

3. **Simple methods competitive**: Gray World (5.97°) and White Patch (5.41°) achieve near-baseline performance with minimal computation

**Main insight**: Combining spatial illumination estimation (Retinex) with color physics (spectral ratios) is challenging because they model different aspects of the problem. When neural spectral ratio predictions are accurate, **simple direct application (color correction) beats complex spatial integration**.

The hybrid approach (learned ISD + physics-based processing) provides practical efficiency (~4s/image), with SR Color Correction offering the best accuracy-speed trade-off for illumination correction tasks.

---

## 6. Running the Code

**Google Colab** (Recommended):
```
1. Open run_on_colab.ipynb in Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells (model auto-downloads from Drive)
4. Download results as .tar.gz
```

**Repository**: `github.com/oelmady/Spectral_Ratio_Illumination_Demo`

**Key Files**:
- `algorithms/retinex.py` - Core algorithms
- `model/unet_models2.py` - Neural architecture
- `scripts/run_batch.py` - Batch processing
- `run_on_colab.ipynb` - Colab execution notebook

**Dependencies**: PyTorch 2.0+, OpenCV 4.0+, NumPy (see `requirements.txt`)

---

## References

1. Land & McCann (1971). Lightness and retinex theory. *JOSA*
2. Finlayson et al. (1994). Spectral sharpening for color constancy. *JOSA A*
3. Grosse et al. (2009). Ground truth dataset for intrinsic images. *ICCV*
4. Barron & Malik (2015). Shape, illumination, and reflectance from shading. *TPAMI*