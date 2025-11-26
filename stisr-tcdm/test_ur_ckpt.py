import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from guided_diffusion.script_util import sr_create_model_and_diffusion, sr_model_and_diffusion_defaults
from text_recognition.model.crnn_numeric import *

# Load the checkpoint
ckpt_path = "/mnt/disk1/ayesha_train/stisr-tcdm/text_recognition/ckpt/best_checkpoint_pytorch.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(len([k for k in ckpt.keys() if "up" in k or "down" in k]))

"""# Check if training args were saved
if 'args' in ckpt:
    args = ckpt['args']
    small_size = getattr(args, 'small_size', None)
    large_size = getattr(args, 'large_size', None)
    print(f"Model expects LR input size: {small_size}, HR output size: {large_size}")
else:
    print("No training args found in checkpoint. You may need to check the training script or logs.")

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
"""
"""
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("Keys in checkpoint:")
for k, v in state_dict.items():
    print(k, v.shape)"""



# Load .npz file
data_path = "diff_samples/hr_samples/1000_samples_1.npz"
data = np.load(data_path)
for k in data.files:
    print(k, data[k].shape, data[k].dtype)
images = data["arr_0"]
print("Min:", images.min(), "Max:", images.max(), "Mean:", images.mean(), "Std:", images.std())
print("Generated samples:", images.shape)


# Create output folder
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

# Save the first 5 images
for i in range(10):
    plt.imshow(images[i])
    plt.axis("off")
    
    # Save figure
    save_path = os.path.join(save_dir, f"image_{i}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # close figure to free memory

    print(f"Saved: {save_path}")