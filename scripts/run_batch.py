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
from pathlib import Path

from model.unet_models2 import ResNet50UNet
from run import ISDMapEstimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-model', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='model/UNET_run_x10_01_last_model.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--image', type=str, default=None, help='Process a single image name (stem only)')
    parser.add_argument('--retinex', action='store_true', help='Run spectral-ratio constrained Retinex and save corrected image')
    parser.add_argument('--sr-correct', action='store_true', help='Apply spectral-ratio color correction (simple shift)')
    parser.add_argument('--distance', type=float, default=1.0, help='Distance in log-space for sr-correct')
    return parser.parse_args()
from algorithms.retinex import spectral_ratio_retinex, apply_spectral_ratio_color_correction


def main():
    args = parse_args()
    use_model = args.use_model
    checkpoint = args.checkpoint
    device = args.device
    image_name = args.image

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
            corrected, illum = spectral_ratio_retinex(image, out_map, iterations=5, sigma=15, anchor=None)
            # Save corrected as 16-bit (scale back)
            corr_u16 = np.clip(corrected, 0, 65535).astype(np.uint16)
            corr_out = out_dir / f"{stem}_retinex_corrected.tiff"
            cv2.imwrite(str(corr_out), corr_u16)
            print(f"Saved Retinex corrected: {corr_out}")

        # Optional: simple spectral-ratio color correction (shift along SR)
        if args.sr_correct:
            corrected2 = apply_spectral_ratio_color_correction(image, out_map, distance=args.distance)
            corr2_u16 = np.clip(corrected2, 0, 65535).astype(np.uint16)
            corr2_out = out_dir / f"{stem}_sr_shifted.tiff"
            cv2.imwrite(str(corr2_out), corr2_u16)
            print(f"Saved SR-shifted image: {corr2_out}")


if __name__ == '__main__':
    main()
