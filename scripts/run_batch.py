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
    return parser.parse_args()


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


if __name__ == '__main__':
    main()
