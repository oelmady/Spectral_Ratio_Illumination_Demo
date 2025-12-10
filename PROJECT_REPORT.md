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
for iteration in range(5):
    I_candidate = GaussianBlur(log(img))
    delta = I_candidate - I
    # KEY: Project delta onto spectral direction
    delta_proj = (delta · sr_unit) * sr_unit  
    I = I + delta_proj
```
**Benefit**: Illumination updates **constrained to physical directions** → better color preservation

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

### Qualitative Findings

**SR-Constrained vs Baseline**:
- ✅ **Better color preservation**: Materials maintain consistent colors across illumination changes
- ✅ **Physically plausible**: Illumination follows expected spectral directions (sun/sky)
- ✅ **Fewer artifacts**: Reduced color halos and unnatural hue shifts
- ✅ **Natural appearance**: Shadow→light transitions look realistic

**Baseline Retinex Issues**:
- ❌ Color shifts in shadow regions (materials appear to change hue)
- ❌ Over-smoothing with large blur (loses detail)
- ❌ Chromaticity preservation creates unrealistic colors

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
1. **Physically-constrained Retinex**: Projecting illumination updates onto spectral directions produces more accurate decomposition
2. **Hybrid architecture**: Neural ISD prediction + classical Retinex combines learning and physics
3. **Practical pipeline**: Complete system with Google Colab notebook for reproducibility

### Limitations
- **ISD quality dependence**: Bad spectral direction predictions → bad results
- **Two-illuminant assumption**: Trained for sun/sky; may not generalize to complex indoor scenes
- **No ground truth evaluation**: Qualitative assessment only (lack of reflectance GT)

### Future Work
- End-to-end learning (train network to output SR-constrained decomposition directly)
- Multi-illuminant extension (>2 light sources)
- Automatic parameter selection from image analysis
- Real-time GPU implementation for video

---

## 5. Conclusion

SR-constrained Retinex achieves **more physically plausible and visually accurate** intrinsic decomposition than baseline chromaticity-preserving methods. By constraining illumination updates to learned spectral directions, the method:
- Preserves material colors across illumination changes
- Reduces artifacts (halos, color shifts)
- Respects physics of light transport

The hybrid approach (learned ISD + physics-based processing) provides high quality with practical efficiency (~4s/image).

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