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

    for _ in range(iterations):
        I_candidate = _gaussian_blur_per_channel(log_img, sigma)
        delta = I_candidate - I

        # Project delta onto SR unit vector per-pixel
        dot = np.einsum('ijk,ijk->ij', delta, sr_unit)
        dot = dot[:, :, np.newaxis]
        delta_proj = dot * sr_unit

        I = I + delta_proj

    # Reflectance estimate
    R = log_img - I

    if anchor is None:
        anchor = float(np.percentile(I, 95))

    corrected_log = R + anchor
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
