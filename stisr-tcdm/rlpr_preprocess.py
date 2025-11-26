import numpy as np
import torch
import cv2

def resize_and_prepare(npz_path, target_height=32, target_width=128):
    """
    Loads an .npz file and resizes the images to target dimensions.
    Converts to torch.Tensor in NCHW format, float32 normalized to [0,1].
    
    Args:
        npz_path (str): Path to the .npz file.
        target_height (int): Desired height.
        target_width (int): Desired width.
    
    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W), float32 [0,1].
    """
    data = np.load(npz_path)['arr_0']  # shape: (N, H, W, C)
    resized_list = []
    for img in data:
        # Resize using cv2
        resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_list.append(resized)
    
    resized_arr = np.stack(resized_list, axis=0)  # (N, H, W, C)
    
    # Convert to torch tensor, permute to NCHW, normalize to [0,1]
    tensor = torch.from_numpy(resized_arr).permute(0, 3, 1, 2).float() / 255.0
    return tensor

if __name__ == "__main__":
    # Example usage
    lr_tensor = resize_and_prepare("RLPR_LR.npz")
    hr_tensor = resize_and_prepare("RLPR_HR.npz", target_height=64, target_width=128)
    print("LR Tensor:", lr_tensor.shape, lr_tensor.dtype)
    print("HR Tensor:", hr_tensor.shape, hr_tensor.dtype)
