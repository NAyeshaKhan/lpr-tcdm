#!/usr/bin/env python3
"""
Resize all MR .npz samples from their original size (32x128)
to the model-required size: 128x64 (H=128, W=64)
without rotation.

Outputs go to:
    diff_samples/mr_samples/resized-128x64/
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image


def resize_npz_file(input_path, output_path, target_h=128, target_w=64):
    print(f"\n=== Processing {input_path} ===")

    # Load .npz
    data = np.load(input_path)
    key = list(data.keys())[0]
    images = data[key]  # shape (N, H, W, 3)

    N, H, W, C = images.shape
    print(f"Loaded: shape={images.shape}, dtype={images.dtype}, min/max={images.min()}/{images.max()}")

    # Prepare output container
    resized = np.zeros((N, target_h, target_w, C), dtype=images.dtype)

    for i in range(N):
        img = Image.fromarray(images[i])
        # PIL takes (width, height)
        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        resized[i] = np.array(img_resized)

        if i == 0:
            print(f"First sample resized: {images[i].shape} → {resized[i].shape}")

        if (i + 1) % 200 == 0:
            print(f"  Resized {i+1}/{N} images")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, resized)

    print(f"Saved to: {output_path}")
    print(f"Final array shape: {resized.shape}")
    print(f"Final min/max: {resized.min()}/{resized.max()}")
    print("=== Done ===")


def batch_resize(input_dir="./diff_samples/mr_samples", output_dir="./diff_samples/mr_samples/res-128x64"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} files. Saving resized versions into {output_dir}")

    for f in npz_files:
        out_path = output_dir / f.name
        resize_npz_file(str(f), str(out_path))


if __name__ == "__main__":
    batch_resize()
