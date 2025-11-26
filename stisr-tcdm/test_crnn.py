import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from text_recognition.recognizer_init import CRNN_init
from text_recognition.utils import get_string_crnn

# =====================
# Load the HR-generated image
# =====================
npz_path = './diff_samples/hr_samples/RLPR_HR_generated.npz'

# Load first image
images = np.load(npz_path)['arr_0']  # shape [N, H, W, C]
sample_img = images[5]  # pick the first image

# =====================
# Visualize the image
# =====================
plt.imshow(sample_img.astype(np.uint8))
plt.axis('off')
plt.title("Input HR-generated Image")
plt.show()

# =====================
# Convert to torch tensor and normalize [-1,1]
# =====================
img_tensor = torch.from_numpy(sample_img).float() / 127.5 - 1.0  # [H, W, C]
img_tensor = img_tensor.permute(2, 0, 1)[None]  # [1, C, H, W]

# =====================
# Resize for CRNN (target height = 32)
# =====================
target_h = 32
C, H, W = img_tensor.shape[1:]
new_W = int(W * target_h / H)
img_resized = F.interpolate(img_tensor, size=(target_h, new_W), mode='bilinear', align_corners=False)

# =====================
# Load CRNN
# =====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
crnn = CRNN_init('./text_recognition/ckpt/crnn.pth').to(device)
crnn.eval()

# Move image to device
img_resized = img_resized.to(device)

# =====================
# Run CRNN
# =====================
with torch.no_grad():
    output, _ = crnn(img_resized[:, :3, :, :])  # take first 3 channels
    pred_text = get_string_crnn(output)

print("Predicted text:", pred_text[0])
