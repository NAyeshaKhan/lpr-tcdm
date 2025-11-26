#Modified, cos OG had dummy code that wasn't working???
"""
Generate a large batch of samples from the synthesizer model,
using TextZoom HR images to produce MR text-image pairs.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data_textzoom_test
from guided_diffusion.image_datasets import TextZoomDataset_SR_Test

# Initialize single-GPU distributed if needed
import torch
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=0,
        world_size=1,
    )


def main():
    args = create_argparser().parse_args()
    out_path = args.out_path
    logger.log(f"out filename: {out_path}")

    out_root = os.path.dirname(out_path)
    os.makedirs(out_root, exist_ok=True)

    dist_util.setup_dist(args.gpu)
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.batch_size, args.num_samples, args.iter, args.level)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        if "low_res" in model_kwargs:
            model_kwargs["low_res"] = model_kwargs["low_res"].to(dist_util.dev())

        if not args.use_ddim:
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, 32, 128),  # MR image size
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
        else:
            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 3, 32, 128),
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        # Gather samples across GPUs if distributed
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample)
        else:
            all_samples = [sample]

        for s in all_samples:
            all_images.append(s.cpu().numpy())

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(batch_size, n_samples, iter, level="easy"):
    """
    Load HR images from TextZoom and prepare MR text-image batches for synthesizer.
    Supports 'easy', 'medium', 'hard' difficulty levels.
    
    Yields:
        dict: {
            'low_res': Tensor of shape (B, 3, H, W),
            'gt_text_label': list of strings
        }
    """
    # Load the dataset for the requested difficulty level
    hr_dataset = TextZoomDataset_SR_Test(level=level)
    if len(hr_dataset) == 0:
        raise RuntimeError(f"No images found for level {level}")

    # Get HR images and their corresponding text labels
    hr_images, hr_texts = hr_dataset.get_imgarr()  # get all images/texts
    hr_images = hr_images[:n_samples]
    hr_texts = hr_texts[:n_samples]
    # returns list/np.array

    # Save ground-truth text labels for this iteration
    words_filename = f'words_list_{iter}.txt'
    out_root = os.path.dirname(os.path.abspath(words_filename))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, words_filename), 'w') as f:
        for word in hr_texts:
            f.write(word + "\n")

    # Distributed setup
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    buffer_imgs = []
    buffer_texts = []

    while True:
        # Split work across ranks
        for i in range(rank, len(hr_images), world_size):
            buffer_imgs.append(hr_images[i])
            buffer_texts.append(hr_texts[i])

            if len(buffer_imgs) == batch_size:
                # Convert images to Tensor, normalize, permute
                batch_imgs = th.from_numpy(np.stack(buffer_imgs)).float()  # (B,H,W,C)
                batch_imgs = batch_imgs / 127.5 - 1.0  # [-1,1]
                batch_imgs = batch_imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

                yield {
                    "low_res": batch_imgs.to(th.device("cuda" if th.cuda.is_available() else "cpu")),
                    "gt_text_label": buffer_texts
                }

                buffer_imgs, buffer_texts = [], []

def create_argparser():
    defaults = dict(
        gpu=0,
        iter=1,
        txt_rec_model="",
        out_path="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        level="easy",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
