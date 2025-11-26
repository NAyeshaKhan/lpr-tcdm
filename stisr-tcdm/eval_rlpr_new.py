import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

lr   = np.load('diff_samples/RLPR_preprocessed/RLPR_LR.npz')['arr_0']
hr   = np.load('diff_samples/RLPR_preprocessed/RLPR_HR.npz')['arr_0']
pred = np.load('diff_samples/hr_samples/RLPR_HR_generated.npz')['arr_0']


print("LR shape:", lr.shape)
print("HR shape:", hr.shape)
print("Pred shape:", pred.shape)

# Resize HR to match pred resolution
h, w = pred.shape[1], pred.shape[2]

hr_resized = np.zeros_like(pred)
for i in range(len(hr)):
    hr_resized[i] = cv2.resize(hr[i], (w, h))

print("New HR shape:", hr_resized.shape)

psnr_list = []
ssim_list = []

for i in range(len(hr)):
    psnr_list.append(peak_signal_noise_ratio(hr_resized[i], pred[i], data_range=255))
    ssim_list.append(structural_similarity(hr_resized[i], pred[i], channel_axis=2))

print("PSNR:", np.mean(psnr_list))
print("SSIM:", np.mean(ssim_list))
