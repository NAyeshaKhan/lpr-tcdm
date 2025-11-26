#Currently working debug code
from tqdm import tqdm
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from text_recognition.recognizer_init import *
from text_recognition.utils import *

# =====================
# Dataset
# =====================
class RLPRDataset_Rec_Test(Dataset):
    """
    RLPR evaluation dataset with LR, SR, HR images and labels.
    """
    def __init__(self, lr_path, sr_path, hr_path, labels_path):
        # Load images
        self.lr_images = np.load(lr_path)["arr_0"]
        self.sr_images = np.load(sr_path)["arr_0"]
        self.hr_images = np.load(hr_path)["arr_0"]

        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        assert len(self.lr_images) == len(self.sr_images) == len(self.hr_images) == len(self.labels), \
            f"All datasets must have same length: {len(self.lr_images)}, {len(self.sr_images)}, {len(self.hr_images)}, {len(self.labels)}"

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr_images[idx]).float() / 127.5 - 1.0
        sr = torch.from_numpy(self.sr_images[idx]).float() / 127.5 - 1.0
        hr = torch.from_numpy(self.hr_images[idx]).float() / 127.5 - 1.0

        # Ensure (C, H, W)
        lr = lr.permute(2, 0, 1)
        sr = sr.permute(2, 0, 1)
        hr = hr.permute(2, 0, 1)

        label = self.labels[idx]
        return hr, lr, sr, label

# =====================
# Data loader
# =====================
def get_test_data_w_sr(lr_path, sr_path, hr_path, labels_path, batch_size):
    dataset = RLPRDataset_Rec_Test(lr_path, sr_path, hr_path, labels_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, drop_last=False
    )
    return dataset, loader

# Numeric string filter
def str_filt_numeric(s): 
    return ''.join([c for c in s if c.isdigit()])

# =====================
# MORAN preprocessing (width fix)
# =====================
def prep_moran(x, target_h=32, min_width=16):
    # Convert to grayscale and replicate to 3 channels
    x_gray = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # [B,3,H,W]

    B, C, H, W = x_gray.shape
    # Compute new width proportionally to target height, ensuring at least min_width
    new_W = max(int(W * target_h / H), min_width)

    # Resize to (target_h, new_W)
    x_resized = F.interpolate(x_gray, size=(target_h, new_W), mode='bilinear', align_corners=False)
    return x_resized

# =====================
# Evaluation
# =====================
def main(lr_path, sr_path, hr_path, labels_path, batch_size=64):
    _, loader = get_test_data_w_sr(lr_path, sr_path, hr_path, labels_path, batch_size)

    # Initialize recognizers
    crnn = CRNN_init('./text_recognition/ckpt/crnn.pth').cuda()
    aster, aster_info = Aster_init("./text_recognition/ckpt/aster_demo.pth")
    aster = aster.cuda()
    moran = MORAN_init('./text_recognition/ckpt/demo.pth').cuda()

    return eval_w_sr(loader, crnn, aster, aster_info, moran)

def eval_w_sr(val_loader, crnn, aster, aster_info, moran):
    crnn.eval()
    aster.eval()
    moran.eval()

    # =====================
    # Initialize counters
    # =====================
    total_chars = 0
    correct_chars_sr = correct_chars_lr = correct_chars_hr = 0
    correct_chars_sr_as = correct_chars_lr_as = correct_chars_hr_as = 0
    correct_chars_sr_mo = correct_chars_lr_mo = correct_chars_hr_mo = 0

    cal_ssim = SSIM()
    cal_psnr = calculate_psnr

    ssim_list = []
    psnr_list = []

    target_h = 32
    filter_mode = 'lower'

    # =====================
    # Character-level helper
    # =====================
    def char_accuracy(pred, label, filter_mode='lower'):
        pred_f = str_filt(pred, filter_mode)
        label_f = str_filt(label, filter_mode)
        min_len = min(len(pred_f), len(label_f))
        correct_chars = sum([pred_f[i] == label_f[i] for i in range(min_len)])
        total_chars_local = max(len(label_f), 1)  # avoid division by zero
        return correct_chars, total_chars_local

    # =====================
    # Main loop
    # =====================
    for data in tqdm(val_loader):
        images_hr, images_lr, images_sr, label_strs = data
        batch_size = images_lr.shape[0]

        images_lr = images_lr.cuda()
        images_sr = images_sr.cuda()
        images_hr = images_hr.cuda()

        # -------------------------
        # Resize images for CRNN/ASTER
        # -------------------------
        images_lr_crnn = F.interpolate(images_lr, size=(target_h, images_lr.shape[3]), mode='bilinear', align_corners=False)
        images_sr_crnn = F.interpolate(images_sr, size=(target_h, images_sr.shape[3]), mode='bilinear', align_corners=False)
        images_hr_crnn = F.interpolate(images_hr, size=(target_h, images_hr.shape[3]), mode='bilinear', align_corners=False)

        # -------------------------
        # CRNN predictions
        # -------------------------
        test_output_lr, _ = crnn(images_lr_crnn[:, :3, :, :])
        test_output_hr, _ = crnn(images_hr_crnn[:, :3, :, :])
        test_output_sr, _ = crnn(images_sr_crnn[:, :3, :, :])

        # -------------------------
        # ASTER predictions
        # -------------------------
        test_output_lr_aster = aster(images_lr_crnn[:, :3, :, :] * 2 - 1)
        test_output_hr_aster = aster(images_hr_crnn[:, :3, :, :] * 2 - 1)
        test_output_sr_aster = aster(images_sr_crnn[:, :3, :, :] * 2 - 1)

        # -------------------------
        # MORAN predictions (with prep)
        # -------------------------
        images_lr_mo = prep_moran(images_lr)
        images_sr_mo = prep_moran(images_sr)
        images_hr_mo = prep_moran(images_hr)

        def moran_predict(x):
            x_parsed, length, text, text_rev, converter = parse_moran_data(x)
            out, _ = moran(x_parsed, length, text, text_rev, test=True)
            _, preds = out.max(1)
            sim_preds = converter.decode(preds.data, length.data)
            return [p.split('$')[0].replace(" ", "") for p in sim_preds]

        predict_lr_mo = moran_predict(images_lr_mo)
        predict_sr_mo = moran_predict(images_sr_mo)
        predict_hr_mo = moran_predict(images_hr_mo)

        # -------------------------
        # Metrics (SSIM/PSNR)
        # -------------------------
        images_sr_resized = F.interpolate(images_sr, size=images_hr.shape[2:], mode='bilinear', align_corners=False)
        for b in range(batch_size):
            ssim_list.append(cal_ssim(images_sr_resized[b][None], images_hr[b][None]).item())
            psnr_list.append(cal_psnr(images_sr_resized[b][None], images_hr[b][None]))

        # -------------------------
        # Recognition
        # -------------------------
        predict_result_lr = get_string_crnn(test_output_lr)
        predict_result_hr = get_string_crnn(test_output_hr)
        predict_result_sr = get_string_crnn(test_output_sr)

        predict_result_lr_aster = get_string_aster(test_output_lr_aster, aster_info)
        predict_result_hr_aster = get_string_aster(test_output_hr_aster, aster_info)
        predict_result_sr_aster = get_string_aster(test_output_sr_aster, aster_info)

        # -------------------------
        # Character-level accuracy
        # -------------------------
        for b in range(batch_size):
            label = label_strs[b]

            c, t = char_accuracy(predict_result_sr[b], label, filter_mode)
            correct_chars_sr += c
            total_chars += t

            c, _ = char_accuracy(predict_result_lr[b], label, filter_mode)
            correct_chars_lr += c

            c, _ = char_accuracy(predict_result_hr[b], label, filter_mode)
            correct_chars_hr += c

            c, _ = char_accuracy(predict_result_sr_aster[b], label, filter_mode)
            correct_chars_sr_as += c

            c, _ = char_accuracy(predict_result_lr_aster[b], label, filter_mode)
            correct_chars_lr_as += c

            c, _ = char_accuracy(predict_result_hr_aster[b], label, filter_mode)
            correct_chars_hr_as += c

            c, _ = char_accuracy(predict_sr_mo[b], label, filter_mode)
            correct_chars_sr_mo += c

            c, _ = char_accuracy(predict_lr_mo[b], label, filter_mode)
            correct_chars_lr_mo += c

            c, _ = char_accuracy(predict_hr_mo[b], label, filter_mode)
            correct_chars_hr_mo += c

        torch.cuda.empty_cache()

    # =====================
    # Character-level accuracy
    # =====================
    accuracy = round(correct_chars_sr / total_chars, 4)
    accuracy_lr = round(correct_chars_lr / total_chars, 4)
    accuracy_hr = round(correct_chars_hr / total_chars, 4)
    accuracy_as = round(correct_chars_sr_as / total_chars, 4)
    accuracy_lr_as = round(correct_chars_lr_as / total_chars, 4)
    accuracy_hr_as = round(correct_chars_hr_as / total_chars, 4)
    accuracy_mo = round(correct_chars_sr_mo / total_chars, 4)
    accuracy_lr_mo = round(correct_chars_lr_mo / total_chars, 4)
    accuracy_hr_mo = round(correct_chars_hr_mo / total_chars, 4)
    ssim = round(float(sum(ssim_list) / len(ssim_list)), 6)
    psnr = round(float(sum(psnr_list) / len(psnr_list)), 6)

    print("===== Evaluation results (Character-level) =====")
    print('sr_accuracy: %.2f%%' % (accuracy * 100))
    print('lr_accuracy: %.2f%%' % (accuracy_lr * 100))
    print('hr_accuracy: %.2f%%' % (accuracy_hr * 100))
    print('sr_accuracy aster: %.2f%%' % (accuracy_as * 100))
    print('lr_accuracy aster: %.2f%%' % (accuracy_lr_as * 100))
    print('hr_accuracy aster: %.2f%%' % (accuracy_hr_as * 100))
    print('sr_accuracy moran: %.2f%%' % (accuracy_mo * 100))
    print('lr_accuracy moran: %.2f%%' % (accuracy_lr_mo * 100))
    print('hr_accuracy moran: %.2f%%' % (accuracy_hr_mo * 100))
    print('ssim: %.6f' % ssim)
    print('psnr: %.6f' % psnr)
    print("=====================================================")

    return {
        'sr_accuracy': accuracy * 100,
        'lr_accuracy': accuracy_lr * 100,
        'hr_accuracy': accuracy_hr * 100,
        'ssim': ssim,
        'psnr': psnr
    }


# =====================
# Main
# =====================
if __name__ == '__main__':
    lr_path = './diff_samples/RLPR_preprocessed/RLPR_LR.npz'
    sr_path = './diff_samples/hr_samples/RLPR_HR_generated.npz'
    hr_path = './diff_samples/RLPR_preprocessed/RLPR_HR.npz' 
    labels_path = './diff_samples/RLPR_preprocessed/words_list_1.txt'

    results = main(lr_path, sr_path, hr_path, labels_path, batch_size=40)

    # --------------------------
    # Save results to JSON
    # --------------------------
    save_path = 'rlpr_eval_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n===== Final RLPR Character-Level Evaluation Results =====")
    print(results)
    print(f"Results saved to {save_path}")

