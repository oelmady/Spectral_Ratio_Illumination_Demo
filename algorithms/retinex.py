# Omar Elmady 
# Wednesday, Dec 11
# CS 7180 
import cv2
import numpy as np


def baseline_retinex(image, iterations=5, sigma=15, anchor=None):
    """
    Standard recursive Retinex algorithm (McCann-Sobel style).
    
    This is the baseline that preserves chromaticity - used for comparison
    against the SR-constrained version.
    
    Parameters:
    - image: uint16 RGB image (H,W,3)
    - iterations: number of refinement iterations
    - sigma: gaussian blur sigma for illumination estimation
    - anchor: optional scalar log-illumination anchor
    
    Returns:
    - corrected_linear: float32 linear RGB image
    - illumination: final illumination estimate in log-space
    """
    # Convert to float32 log-space
    log_img = np.zeros_like(image, dtype=np.float32)
    mask = image > 0
    log_img[mask] = np.log(image[mask])
    
    # Initialize illumination with Gaussian blur
    I = _gaussian_blur_per_channel(log_img, sigma)
    
    # Iteratively refine illumination (standard Retinex - no SR constraint)
    for _ in range(iterations):
        I = _gaussian_blur_per_channel(log_img - (_gaussian_blur_per_channel(log_img, sigma) - I), sigma)
    
    # Reflectance estimate
    R = log_img - I
    
    if anchor is None:
        anchor = float(np.percentile(I, 95))
    
    corrected_log = R + anchor
    corrected_linear = np.exp(corrected_log).astype(np.float32)
    
    return corrected_linear, I


def _gaussian_blur_per_channel(log_img, sigma):
    # kernel size from sigma
    ksize = max(3, int(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1
    blurred = np.zeros_like(log_img)
    for c in range(log_img.shape[2]):
        blurred[:, :, c] = cv2.GaussianBlur(log_img[:, :, c], (ksize, ksize), sigma)
    return blurred


def normalize_sr_map(sr_map):
    """Normalize SR map to unit vectors per pixel."""
    norm = np.linalg.norm(sr_map, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    return sr_map / norm


def spectral_ratio_retinex(image, sr_map, iterations=5, sigma=15, anchor=None):
    """
    Lightweight spectral-ratio constrained Retinex-like routine.

    This implements a simple iterative illumination estimation where each
    illumination update is projected onto the per-pixel spectral-ratio direction.

    Parameters:
    - image: uint16 RGB image (H,W,3)
    - sr_map: float32 normalized ISD map (H,W,3)
    - iterations: number of refinement iterations
    - sigma: gaussian blur sigma used for coarse illumination estimate
    - anchor: optional scalar log-illumination anchor applied to corrected image

    Returns:
    - corrected_linear: float32 linear RGB image (same scale as input)
    - illumination: final illumination estimate in log-space
    """
    # Convert to float32 log-space
    log_img = np.zeros_like(image, dtype=np.float32)
    mask = image > 0
    log_img[mask] = np.log(image[mask])

    # Initialize illumination with Gaussian blur of log image
    I = _gaussian_blur_per_channel(log_img, sigma)

    sr_unit = normalize_sr_map(sr_map.astype(np.float32))

    # Standard Retinex iterations (unconstrained)
    for _ in range(iterations):
        I = _gaussian_blur_per_channel(log_img - (_gaussian_blur_per_channel(log_img, sigma) - I), sigma)

    # Reflectance estimate (standard Retinex)
    R = log_img - I
    
    # SR constraint: Keep reflectance chromaticity, but shift magnitude along SR
    # This preserves material color while adjusting brightness via SR direction
    # Project R onto SR to get how much to shift along SR
    dot = np.einsum('ijk,ijk->ij', R, sr_unit)
    dot = dot[:, :, np.newaxis]
    R_sr_component = dot * sr_unit
    
    # Blend between original reflectance and SR-aligned version
    alpha = 0.8  # Keep 80% of original reflectance chromaticity
    R_corrected = alpha * R + (1 - alpha) * R_sr_component

    if anchor is None:
        anchor = float(np.percentile(I, 95))

    corrected_log = R_corrected + anchor
    corrected_linear = np.exp(corrected_log).astype(np.float32)

    return corrected_linear, I


def apply_spectral_ratio_color_correction(image, sr_map, distance=1.0, mask=None):
    """
    Shift pixels in log-space along the spectral-ratio vectors by `distance`.

    Parameters:
    - image: uint16 RGB image (H,W,3)
    - sr_map: float32 unit SR map (H,W,3)
    - distance: scalar distance in log-space to shift (positive brightens along SR)
    - mask: optional boolean mask (H,W) to limit where correction is applied

    Returns:
    - corrected_linear: float32 linear RGB image
    """
    log_img = np.zeros_like(image, dtype=np.float32)
    mask_pixels = image > 0
    log_img[mask_pixels] = np.log(image[mask_pixels])

    sr_unit = normalize_sr_map(sr_map.astype(np.float32))

    shifted = log_img.copy()
    if mask is not None:
        ys, xs = np.where(mask)
        shifted[ys, xs, :] += distance * sr_unit[ys, xs, :]
    else:
        shifted += distance * sr_unit

    corrected_linear = np.exp(shifted).astype(np.float32)
    return corrected_linear


def gray_world_correction(image):
    """
    Gray World color constancy algorithm.
    Assumes the average color in the scene should be gray.
    
    Parameters:
    - image: uint16 RGB image (H,W,3)
    
    Returns:
    - corrected_linear: float32 linear RGB image
    """
    img_float = image.astype(np.float32)
    
    # Compute mean of each channel
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    # Gray value (average of channel means)
    gray = (mean_r + mean_g + mean_b) / 3.0
    
    # Scale each channel
    corrected = np.zeros_like(img_float)
    corrected[:, :, 0] = img_float[:, :, 0] * (gray / mean_r) if mean_r > 0 else img_float[:, :, 0]
    corrected[:, :, 1] = img_float[:, :, 1] * (gray / mean_g) if mean_g > 0 else img_float[:, :, 1]
    corrected[:, :, 2] = img_float[:, :, 2] * (gray / mean_b) if mean_b > 0 else img_float[:, :, 2]
    
    return corrected.astype(np.float32)


def white_patch_correction(image, percentile=99):
    """
    White Patch (Max RGB) color constancy algorithm.
    Assumes the brightest pixel in the scene should be white.
    
    Parameters:
    - image: uint16 RGB image (H,W,3)
    - percentile: which percentile to use as "white" (default 99 to avoid outliers)
    
    Returns:
    - corrected_linear: float32 linear RGB image
    """
    img_float = image.astype(np.float32)
    
    # Find the brightest value in each channel (using percentile to avoid outliers)
    max_r = np.percentile(img_float[:, :, 0], percentile)
    max_g = np.percentile(img_float[:, :, 1], percentile)
    max_b = np.percentile(img_float[:, :, 2], percentile)
    
    # Normalize by the brightest channel
    max_overall = max(max_r, max_g, max_b)
    
    # Scale each channel
    corrected = np.zeros_like(img_float)
    corrected[:, :, 0] = img_float[:, :, 0] * (max_overall / max_r) if max_r > 0 else img_float[:, :, 0]
    corrected[:, :, 1] = img_float[:, :, 1] * (max_overall / max_g) if max_g > 0 else img_float[:, :, 1]
    corrected[:, :, 2] = img_float[:, :, 2] * (max_overall / max_b) if max_b > 0 else img_float[:, :, 2]
    
    return corrected.astype(np.float32)


def multiscale_retinex(image, sigmas=[15, 80, 250], anchor=None):
    """
    Multi-Scale Retinex (MSR) algorithm.
    Combines Retinex at multiple scales for better dynamic range compression.
    
    Parameters:
    - image: uint16 RGB image (H,W,3)
    - sigmas: list of Gaussian blur sigma values for different scales
    - anchor: optional scalar log-illumination anchor
    
    Returns:
    - corrected_linear: float32 linear RGB image
    - illumination: final multi-scale illumination estimate in log-space
    """
    # Convert to float32 log-space
    log_img = np.zeros_like(image, dtype=np.float32)
    mask = image > 0
    log_img[mask] = np.log(image[mask])
    
    # Compute multi-scale illumination estimate
    I_multi = np.zeros_like(log_img)
    for sigma in sigmas:
        I_scale = _gaussian_blur_per_channel(log_img, sigma)
        I_multi += I_scale
    
    # Average across scales
    I_multi /= len(sigmas)
    
    # Reflectance estimate
    R = log_img - I_multi
    
    if anchor is None:
        anchor = float(np.percentile(I_multi, 95))
    
    corrected_log = R + anchor
    corrected_linear = np.exp(corrected_log).astype(np.float32)
    
    return corrected_linear, I_multi
