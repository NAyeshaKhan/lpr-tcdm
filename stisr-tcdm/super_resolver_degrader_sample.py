"""
Generate a large batch of samples from a super-resolution model,
given a batch of MR samples from a synthesizer or GT-DIMSS.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    logger.log('out_path: {}'.format(args.out_path))
    logger.log('base_samples: {}'.format(args.base_samples))

    words_list_path = os.path.dirname(args.base_samples)
    logger.log('words_list_path: {}'.format(words_list_path))

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
    data = load_data_for_worker(
        args.base_samples, args.batch_size, words_list_path, args.iter
    )

    logger.log("creating samples...")
    all_images = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs["low_res"] = model_kwargs["low_res"].to(dist_util.dev())

        # Use actual MR dimensions to define HR shape
        _, _, H, W = model_kwargs["low_res"].shape
        upscale_factor = args.large_size // H  # compute approximate scaling
        """hr_H = H * upscale_factor
        hr_W = W * upscale_factor"""
        hr_H = 64
        hr_W = 128
        shape = (args.batch_size, 3, hr_H, hr_W)

        # Generate super-resolved samples
        if not args.use_ddim:
            sample = diffusion.p_sample_loop(
                model,
                shape,
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
        else:
            sample = diffusion.ddim_sample_loop(
                model,
                shape,
                clip_denoised=args.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )

        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

        # Gather across GPUs if distributed
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample)
        else:
            all_samples = [sample]

        for s in all_samples:
            all_images.append(s.cpu().numpy())

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Concatenate and trim
    arr = np.concatenate(all_images, axis=0)[: args.num_samples]

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.log(f"saving to {args.out_path}")
        np.savez(args.out_path, arr)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, words_list_path, iter):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
    print(f"Loaded {len(image_arr)} images from {base_samples}")

    words_filename = f'words_list_{iter}.txt'
    words_file_path = os.path.join(words_list_path, words_filename)
    if not os.path.exists(words_file_path):
        raise FileNotFoundError(f"Words list file not found: {words_file_path}")

    with open(words_file_path, 'r') as f:
        words_list = [line.strip() for line in f if line.strip()]

    if len(words_list) != len(image_arr):
        print(f"Warning: number of words ({len(words_list)}) != number of images ({len(image_arr)})")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    num_ranks = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    buffer = []
    word_buffer = []

    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            word_buffer.append(words_list[i])

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)  # (B,C,H,W)

                # Do NOT resize; keep MR original shape
                res = dict(low_res=batch)
                res["gt_text_label"] = word_buffer
                yield res
                buffer, word_buffer = [], []


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
        base_samples="",
        model_path="",
        small_size=64,   # MR / LR size (for logging only)
        large_size=128,   # HR / super-resolved size (used to compute upscale factor)
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    main()
