"""
Train a super-resolution model.
"""
#Modified

import os
os.environ['OPENAI_LOGDIR'] = './ckpt'
import argparse

from text_recognition.recognizer_init import *

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_textzoom, load_data_textzoom_deg
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import os
import torch
import torch.distributed as dist

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
#End Modified

def main():
    args = create_argparser().parse_args()
    #Added
    os.environ['OPENAI_LOGDIR'] = args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    #End Added
    dist_util.setup_dist()
    logger.configure()

    if args.use_gt_dimss:
        text_rec = None
    else:
        text_rec = CRNN_init("./text_recognition/ckpt/crnn.pth")
        text_rec.to(dist_util.dev())
        text_rec.train()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_superres_data(args.batch_size, args.use_degrader)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        text_rec=text_rec,
        max_steps=args.max_steps,
        args=args,
    ).run_loop()
    #Added
    print(f"Max steps: {args.max_steps}")
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    #End Added


def load_superres_data(batch_size, use_degrader):
    if not use_degrader:
        data = load_data_textzoom(batch_size=batch_size)
    else:
        data = load_data_textzoom_deg(batch_size=batch_size)
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        gpu=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_synthesizer=False,
        use_degrader=False,
        max_steps=100000,  #Modified
        ckpt_dir="./ckpt", #Modified
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

"""
Main training script that sets up distributed training, initializes the model and data loaders for the TextZoom dataset, 
optionally loads a text recognizer, and runs the training loop for the diffusion-based text image super-resolution model 
while managing checkpoints and logging.
"""