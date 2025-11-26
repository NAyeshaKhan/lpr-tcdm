import os
import glob
import numpy as np
import shutil

#Heavily modified!!
# Root folder containing the aligned NPZ files and word lists
root = './diff_samples/mr_samples/aligned'

# Folder to save postprocessed files
postprocessed_root = os.path.join(root, 'postprocessed')
os.makedirs(postprocessed_root, exist_ok=True)

# Get all .npz files in the aligned folder
npz_files = sorted(glob.glob(os.path.join(root, '*.npz')))
print(f'Found {len(npz_files)} NPZ files.')

for npz_file in npz_files:
    # Load images
    samples = np.load(npz_file)["arr_0"]
    print(f'File: {os.path.basename(npz_file)} - Loaded {samples.shape[0]} samples.')

    # Save images in postprocessed folder (same filename)
    out_file = os.path.join(postprocessed_root, os.path.basename(npz_file))
    np.savez(out_file, samples)
    print(f'Saved {out_file}')

    # Copy corresponding words_list_*.txt
    base_name = os.path.basename(npz_file).replace('.npz', '')
    word_list_file = os.path.join(root, f'../words_list_{base_name.split("_")[-1]}.txt')
    if os.path.exists(word_list_file):
        shutil.copy(word_list_file, postprocessed_root)
        print(f'Copied {word_list_file} to {postprocessed_root}')
    else:
        print(f'Warning: Word list {word_list_file} not found.')
