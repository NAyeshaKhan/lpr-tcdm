import os
import numpy as np
import cv2
import subprocess
from pathlib import Path

###############################################################################
# CONFIGURATION — CHANGE ONLY IF NEEDED
###############################################################################

RLPR_ROOT = "RLPR"                      # Path to original RLPR dataset
OUT_DIR = "RLPR_outputs"                # Output directory

SUPER_RESOLVER = "super_resolver_degrader_sample.py"
SR_CKPT = "./ckpt/super_resolver/ema_0.9999_100000.pt"

# IMPORTANT: Your model was trained with:
# small_size = 128 (LR width)
# large_size = 64  (LR height)
LR_SIZE = (128, 64)       # (width, height) = (128, 64)
HR_SIZE = (256, 128)      # (width, height) = (256, 128)

GPU = "1"

###############################################################################
# STEP 1 — PREPROCESS RLPR → NPZ
###############################################################################

def preprocess_rlpr(rlpr_root, out_dir):

    print("Preprocessing RLPR dataset...")

    dataset_root = Path(rlpr_root, "Dataset")

    # Sort numerically: sample_001, sample_002, ..., sample_200
    samples = sorted(
        [d for d in dataset_root.glob("sample_*")],
        key=lambda x: int(x.name.split("_")[1])
    )

    lr_list = []
    hr_list = []
    index_list = []
    label_list = []

    # Load labels
    label_path = Path(rlpr_root, "Label", "Labels.txt")
    with open(label_path, "r") as f:
        all_labels = [x.strip() for x in f]

    for sample_dir in samples:
        sample_id = sample_dir.name
        idx = int(sample_id.split("_")[1]) - 1

        # Load LR
        lr_path = sample_dir / "SR_ROI.png"
        lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        if lr_img is None:
            raise RuntimeError(f"LR image missing: {lr_path}")

        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.resize(lr_img, LR_SIZE)  # (width, height)

        # Load HR
        hr_path = sample_dir / "Pseudo_GT.png"
        hr_img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        if hr_img is None:
            raise RuntimeError(f"HR image missing: {hr_path}")

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.resize(hr_img, HR_SIZE)

        lr_list.append(lr_img)
        hr_list.append(hr_img)
        index_list.append(sample_id)
        label_list.append(all_labels[idx])

    # Save NPZ files
    lr_npz = Path(out_dir) / "RLPR_LR.npz"
    hr_npz = Path(out_dir) / "RLPR_HR.npz"
    index_txt = Path(out_dir) / "RLPR_index.txt"
    label_txt = Path(out_dir) / "RLPR_labels.txt"

    np.savez(lr_npz, np.array(lr_list))
    np.savez(hr_npz, np.array(hr_list))

    with open(index_txt, "w") as f:
        for item in index_list:
            f.write(item + "\n")

    with open(label_txt, "w") as f:
        for item in label_list:
            f.write(item + "\n")

    print("Preprocessing complete.")
    return lr_npz, hr_npz, index_txt, label_txt



###############################################################################
# STEP 2 — RUN SUPER-RESOLVER (INFERENCE)
###############################################################################

def run_super_resolver(lr_npz, out_dir):

    print("Running super-resolution inference...")

    num_samples = len(np.load(lr_npz)["arr_0"])
    sr_npz = Path(out_dir) / "RLPR_SR.npz"

    cmd = [
        "python", SUPER_RESOLVER,
        "--kb", "9",
        "--use_gt_dimss", "1",
        "--use_synthesizer", "0",
        "--gpu", GPU,
        "--model_path", SR_CKPT,
        "--num_samples", str(num_samples),
        "--base_samples", str(lr_npz),
        "--large_size", str(LR_SIZE[1]),   # height = 64
        "--small_size", str(LR_SIZE[0]),   # width = 128
        "--batch_size", "50",
        "--timestep_respacing", "250",
        "--out_path", str(sr_npz)
    ]

    subprocess.run(cmd)
    print("Super-resolution complete.")
    return sr_npz



###############################################################################
# STEP 3 — CONVERT RLPR INTO TEXTZOOM-STYLE (FOR EVALUATE.PY)
###############################################################################

def convert_to_textzoom(lr_npz, hr_npz, sr_npz, index_txt, labels_txt, out_dir):

    print("Converting RLPR to TextZoom-style folder...")

    lr = np.load(lr_npz)["arr_0"]
    hr = np.load(hr_npz)["arr_0"]
    sr = np.load(sr_npz)["arr_0"]

    with open(index_txt) as f:
        names = [x.strip() for x in f]
    with open(labels_txt) as f:
        labels = [x.strip() for x in f]

    assert len(lr) == len(hr) == len(sr) == len(names) == len(labels)

    tz_root = Path(out_dir) / "RLPR_processed" / "test"
    (tz_root / "lr").mkdir(parents=True, exist_ok=True)
    (tz_root / "hr").mkdir(exist_ok=True)
    (tz_root / "sr").mkdir(exist_ok=True)

    # Write labels in TextZoom-style
    with open(tz_root / "label.txt", "w") as f:
        for name, label in zip(names, labels):
            f.write(f"{name}.png {label}\n")

    # Write images
    for i, name in enumerate(names):
        fname = name + ".png"
        cv2.imwrite(str(tz_root / "lr" / fname), cv2.cvtColor(lr[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tz_root / "hr" / fname), cv2.cvtColor(hr[i], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tz_root / "sr" / fname), cv2.cvtColor(sr[i], cv2.COLOR_RGB2BGR))

    print("Conversion complete.")
    return tz_root



###############################################################################
# STEP 4 — RUN EVALUATION (PSNR/SSIM/RECOGNITION)
###############################################################################

def run_evaluation(tz_root):

    print("Running evaluation...")

    cmd = [
        "python", "evaluate.py",
        "--tz_root", str(tz_root),
        "--tz_sr_dir", str(tz_root / "sr"),
    ]

    subprocess.run(cmd)

    print("Evaluation complete.")



###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n================ RLPR FULL PIPELINE ================\n")

    # Step 1
    lr_npz, hr_npz, index_txt, labels_txt = preprocess_rlpr(RLPR_ROOT, OUT_DIR)

    # Step 2
    sr_npz = run_super_resolver(lr_npz, OUT_DIR)

    # Step 3
    tz_dir = convert_to_textzoom(lr_npz, hr_npz, sr_npz, index_txt, labels_txt, OUT_DIR)

    # Step 4
    run_evaluation(tz_dir)

    print("\n================ DONE ★ SUCCESS ================\n")
