import numpy as np
from guided_diffusion.image_datasets import TextZoomDataset_SR_Test
import os

# ---------------------------
# Configuration
generated_folder = "diff_samples/mr_samples"   # folder with original .npz files
level = "easy"
output_folder = os.path.join(generated_folder, "aligned")
os.makedirs(output_folder, exist_ok=True)

# List of original .npz files
npz_files = sorted([f for f in os.listdir(generated_folder) if f.endswith(".npz")])

# ---------------------------
# Load GT dataset
dataset = TextZoomDataset_SR_Test(level=level)
num_gt = len(dataset)
print(f"GT samples: {num_gt}")

# Prepare GT images as a list
gt_imgs_list = []
for i in range(num_gt):
    img, _ = dataset[i]  # ignore label
    if isinstance(img, np.ndarray):
        gt_img = img
        if gt_img.shape[0] == 3:  # C,H,W -> H,W,C
            gt_img = gt_img.transpose(1, 2, 0)
    else:
        gt_img = img.numpy().transpose(1, 2, 0)
    gt_imgs_list.append(gt_img)
gt_imgs_list = np.stack(gt_imgs_list)
print(f"GT images stacked: {gt_imgs_list.shape}")

# ---------------------------
# Process each generated .npz file
for npz_file in npz_files:
    print(f"\nProcessing {npz_file} ...")
    
    gen_path = os.path.join(generated_folder, npz_file)
    gen_imgs = np.load(gen_path)["arr_0"]
    num_generated = len(gen_imgs)
    print(f"Generated samples: {gen_imgs.shape}")

    if num_generated > num_gt:
        raise ValueError(f"{npz_file} contains more samples than GT dataset!")

    # Align GT images by index
    gt_aligned = gt_imgs_list[:num_generated]

    # Ensure same value range
    if gen_imgs.max() <= 1.0:
        gen_imgs = (gen_imgs * 255).astype(np.uint8)
    if gt_aligned.max() <= 1.0:
        gt_aligned = (gt_aligned * 255).astype(np.uint8)

    # ---------------------------
    # Save aligned .npz in the aligned folder with the same filename
    gen_out_path = os.path.join(output_folder, npz_file)
    np.savez(gen_out_path, gen_imgs)
    print(f"Saved aligned generated images to {gen_out_path}")

    gt_out_path = os.path.join(output_folder, f"gt_{npz_file}")
    np.savez(gt_out_path, gt_aligned)
    print(f"Saved corresponding GT images to {gt_out_path}")

print("\nAll files processed and saved in the aligned folder.")
