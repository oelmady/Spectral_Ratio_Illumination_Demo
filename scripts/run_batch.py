# Omar Elmady 
# Wednesday, Dec 11
# CS 7180 
"""
Batch runner to produce ISD maps (predicted or annotated) and save visualizations to `results/`.

Usage examples:

python scripts/run_batch.py --use-model --checkpoint model/UNET_run_x10_01_last_model.pth --device cpu
python scripts/run_batch.py --image example_image
"""
import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from model.unet_models2 import ResNet50UNet
from run import ISDMapEstimator
from algorithms.retinex import (
    spectral_ratio_retinex, 
    apply_spectral_ratio_color_correction, 
    baseline_retinex,
    gray_world_correction,
    white_patch_correction,
    multiscale_retinex
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-model', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='model/UNET_run_x10_01_last_model.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--image', type=str, default=None, help='Process a single image name (stem only)')
    parser.add_argument('--retinex', action='store_true', help='Run spectral-ratio constrained Retinex and save corrected image')
    parser.add_argument('--baseline-retinex', action='store_true', help='Run baseline Retinex (no SR constraint) for comparison')
    parser.add_argument('--sr-correct', action='store_true', help='Apply spectral-ratio color correction (simple shift)')
    parser.add_argument('--gray-world', action='store_true', help='Run Gray World color constancy algorithm')
    parser.add_argument('--white-patch', action='store_true', help='Run White Patch (Max RGB) algorithm')
    parser.add_argument('--multiscale-retinex', action='store_true', help='Run Multi-Scale Retinex (MSR)')
    parser.add_argument('--distance', type=float, default=1.0, help='Distance in log-space for sr-correct')
    parser.add_argument('--iterations', type=int, default=5, help='Number of Retinex iterations')
    parser.add_argument('--sigma', type=float, default=15.0, help='Gaussian blur sigma for Retinex')
    parser.add_argument('--compute-metrics', action='store_true', help='Compute quality metrics (SSIM, color constancy)')
    return parser.parse_args()


def compute_color_constancy_error(original, corrected):
    """
    Compute color constancy error as angular error between mean colors.
    Lower is better (colors are more stable).
    """
    # Convert to float and normalize
    orig_float = original.astype(np.float32) / 65535.0
    corr_float = corrected.astype(np.float32) / 65535.0
    
    # Compute mean color for each image
    orig_mean = np.mean(orig_float.reshape(-1, 3), axis=0)
    corr_mean = np.mean(corr_float.reshape(-1, 3), axis=0)
    
    # Normalize to unit vectors
    orig_mean_norm = orig_mean / (np.linalg.norm(orig_mean) + 1e-8)
    corr_mean_norm = corr_mean / (np.linalg.norm(corr_mean) + 1e-8)
    
    # Angular error in degrees
    cos_sim = np.clip(np.dot(orig_mean_norm, corr_mean_norm), -1.0, 1.0)
    angular_error = np.arccos(cos_sim) * 180.0 / np.pi
    
    return angular_error


def compute_ssim_multichannel(img1, img2):
    """
    Compute SSIM for each channel and return average.
    Both images should be uint16.
    """
    # Convert to float [0, 1]
    img1_float = img1.astype(np.float32) / 65535.0
    img2_float = img2.astype(np.float32) / 65535.0
    
    # Compute SSIM per channel
    ssim_scores = []
    for c in range(3):
        score = ssim(img1_float[:, :, c], img2_float[:, :, c], data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)


def main():
    args = parse_args()
    use_model = args.use_model
    checkpoint = args.checkpoint
    device = args.device
    image_name = args.image
    compute_metrics = args.compute_metrics

    image_dir = Path('data/images')
    sr_map_dir = Path('data/sr_maps')
    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ResNet50UNet(in_channels=3, out_channels=3, pretrained=False, checkpoint=None, se_block=True)
    estimator = ISDMapEstimator(model=model, model_path=checkpoint, device=device)

    if image_name:
        stems = [image_name]
    else:
        stems = [p.stem for p in image_dir.glob('*.tif')]
    
    # Only keep last 10 images to save time/storage
    if len(stems) > 10:
        print(f"⚠️ Found {len(stems)} images. Processing only the last 10 to save time.")
        stems = stems[-10:]
    
    # Store metrics for all images
    all_metrics = {}

    for stem in stems:
        img_path = image_dir / f"{stem}.tif"
        sr_path = sr_map_dir / f"{stem}_isd.tiff"

        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store metrics for this image
        image_metrics = {}

        if use_model:
            pred_map = estimator.predict(image)
            out_map = pred_map
        else:
            if not sr_path.exists():
                print(f"SR map not found for {stem}, skipping")
                continue
            sr_map = cv2.imread(str(sr_path), cv2.IMREAD_UNCHANGED)
            sr_map = sr_map.astype(np.float32) / 65535.0
            out_map = sr_map

        # Save as 16-bit TIFF
        out_u16 = np.clip(out_map * 65535.0, 0, 65535).astype(np.uint16)
        tiff_out = out_dir / f"{stem}_isd_pred.tiff"
        cv2.imwrite(str(tiff_out), out_u16)
        print(f"Saved: {tiff_out}")

        # Save visualization
        vis8 = (np.clip(out_map, 0.0, 1.0) * 255.0).astype(np.uint8)
        vis_out = out_dir / f"{stem}_isd_vis.png"
        cv2.imwrite(str(vis_out), vis8)
        print(f"Saved: {vis_out}")

        # Save quick 8-bit reference image
        img8 = (np.clip(image / 256.0, 0, 255)).astype(np.uint8)
        img8_bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        img_out = out_dir / f"{stem}_image_8bit.png"
        cv2.imwrite(str(img_out), img8_bgr)
        print(f"Saved: {img_out}")

        # Optional: run spectral-ratio constrained Retinex
        if args.retinex:
            corrected, illum = spectral_ratio_retinex(image, out_map, iterations=args.iterations, sigma=args.sigma, anchor=None)
            # Save corrected as 16-bit (scale back)
            corr_u16 = np.clip(corrected, 0, 65535).astype(np.uint16)
            corr_out = out_dir / f"{stem}_sr_retinex.tiff"
            cv2.imwrite(str(corr_out), corr_u16)
            print(f"Saved SR-Retinex corrected: {corr_out}")
            
            # Save 8-bit visualization
            corr8 = (np.clip(corrected / 256.0, 0, 255)).astype(np.uint8)
            corr8_bgr = cv2.cvtColor(corr8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_sr_retinex_vis.png"), corr8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr_u16)
                ssim_score = compute_ssim_multichannel(image, corr_u16)
                image_metrics['sr_retinex'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  SR-Retinex - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")

        # Optional: run baseline Retinex (no SR constraint) for comparison
        if args.baseline_retinex:
            corrected_base, illum_base = baseline_retinex(image, iterations=args.iterations, sigma=args.sigma, anchor=None)
            corr_base_u16 = np.clip(corrected_base, 0, 65535).astype(np.uint16)
            corr_base_out = out_dir / f"{stem}_baseline_retinex.tiff"
            cv2.imwrite(str(corr_base_out), corr_base_u16)
            print(f"Saved Baseline Retinex: {corr_base_out}")
            
            # Save 8-bit visualization
            corr_base8 = (np.clip(corrected_base / 256.0, 0, 255)).astype(np.uint8)
            corr_base8_bgr = cv2.cvtColor(corr_base8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_baseline_retinex_vis.png"), corr_base8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr_base_u16)
                ssim_score = compute_ssim_multichannel(image, corr_base_u16)
                image_metrics['baseline_retinex'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  Baseline Retinex - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")

        # Optional: simple spectral-ratio color correction (shift along SR)
        if args.sr_correct:
            corrected2 = apply_spectral_ratio_color_correction(image, out_map, distance=args.distance)
            corr2_u16 = np.clip(corrected2, 0, 65535).astype(np.uint16)
            corr2_out = out_dir / f"{stem}_sr_shifted.tiff"
            cv2.imwrite(str(corr2_out), corr2_u16)
            print(f"Saved SR-shifted image: {corr2_out}")
            
            # Save 8-bit visualization
            corr2_8 = (np.clip(corrected2 / 256.0, 0, 255)).astype(np.uint8)
            corr2_8_bgr = cv2.cvtColor(corr2_8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_sr_shifted_vis.png"), corr2_8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr2_u16)
                ssim_score = compute_ssim_multichannel(image, corr2_u16)
                image_metrics['sr_color_correction'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  SR Color Correction - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")
        
        # Optional: Gray World correction
        if args.gray_world:
            corrected_gw = gray_world_correction(image)
            corr_gw_u16 = np.clip(corrected_gw, 0, 65535).astype(np.uint16)
            corr_gw_out = out_dir / f"{stem}_gray_world.tiff"
            cv2.imwrite(str(corr_gw_out), corr_gw_u16)
            print(f"Saved Gray World: {corr_gw_out}")
            
            # Save 8-bit visualization
            corr_gw8 = (np.clip(corrected_gw / 256.0, 0, 255)).astype(np.uint8)
            corr_gw8_bgr = cv2.cvtColor(corr_gw8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_gray_world_vis.png"), corr_gw8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr_gw_u16)
                ssim_score = compute_ssim_multichannel(image, corr_gw_u16)
                image_metrics['gray_world'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  Gray World - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")
        
        # Optional: White Patch correction
        if args.white_patch:
            corrected_wp = white_patch_correction(image)
            corr_wp_u16 = np.clip(corrected_wp, 0, 65535).astype(np.uint16)
            corr_wp_out = out_dir / f"{stem}_white_patch.tiff"
            cv2.imwrite(str(corr_wp_out), corr_wp_u16)
            print(f"Saved White Patch: {corr_wp_out}")
            
            # Save 8-bit visualization
            corr_wp8 = (np.clip(corrected_wp / 256.0, 0, 255)).astype(np.uint8)
            corr_wp8_bgr = cv2.cvtColor(corr_wp8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_white_patch_vis.png"), corr_wp8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr_wp_u16)
                ssim_score = compute_ssim_multichannel(image, corr_wp_u16)
                image_metrics['white_patch'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  White Patch - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")
        
        # Optional: Multi-Scale Retinex
        if args.multiscale_retinex:
            corrected_msr, illum_msr = multiscale_retinex(image)
            corr_msr_u16 = np.clip(corrected_msr, 0, 65535).astype(np.uint16)
            corr_msr_out = out_dir / f"{stem}_multiscale_retinex.tiff"
            cv2.imwrite(str(corr_msr_out), corr_msr_u16)
            print(f"Saved Multi-Scale Retinex: {corr_msr_out}")
            
            # Save 8-bit visualization
            corr_msr8 = (np.clip(corrected_msr / 256.0, 0, 255)).astype(np.uint8)
            corr_msr8_bgr = cv2.cvtColor(corr_msr8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{stem}_multiscale_retinex_vis.png"), corr_msr8_bgr)
            
            # Compute metrics
            if compute_metrics:
                color_error = compute_color_constancy_error(image, corr_msr_u16)
                ssim_score = compute_ssim_multichannel(image, corr_msr_u16)
                image_metrics['multiscale_retinex'] = {
                    'color_constancy_error_deg': float(color_error),
                    'ssim': float(ssim_score)
                }
                print(f"  Multi-Scale Retinex - Color error: {color_error:.2f}°, SSIM: {ssim_score:.4f}")
        
        # Store metrics for this image
        if compute_metrics and image_metrics:
            all_metrics[stem] = image_metrics
    
    # Save all metrics to JSON
    if compute_metrics and all_metrics:
        metrics_file = out_dir / 'quality_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✓ Quality metrics saved to: {metrics_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("QUALITY METRICS SUMMARY")
        print("="*70)
        print(f"{'Method':<25} {'Avg Color Error (°)':<20} {'Avg SSIM':<15}")
        print("-"*70)
        
        # Compute averages per method
        methods = ['sr_retinex', 'baseline_retinex', 'sr_color_correction']
        method_names = ['SR-Constrained Retinex', 'Baseline Retinex', 'SR Color Correction']
        
        for method, name in zip(methods, method_names):
            color_errors = [m[method]['color_constancy_error_deg'] 
                           for m in all_metrics.values() if method in m]
            ssim_scores = [m[method]['ssim'] 
                          for m in all_metrics.values() if method in m]
            
            if color_errors:
                avg_color = np.mean(color_errors)
                avg_ssim = np.mean(ssim_scores)
                print(f"{name:<25} {avg_color:<20.2f} {avg_ssim:<15.4f}")
        
        print("="*70)
        print("\n💡 Lower color error = better color preservation")
        print("💡 Higher SSIM = better structural similarity to original")
        print("💡 Your method should show LOWER color error than baseline!\n")


if __name__ == '__main__':
    main()
