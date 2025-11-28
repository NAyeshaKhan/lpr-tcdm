import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from guided_diffusion.script_util import sr_create_model_and_diffusion, sr_model_and_diffusion_defaults
from text_recognition.model.crnn_numeric import *

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