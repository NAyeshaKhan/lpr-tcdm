#!/bin/bash -e

# GPU to use
gpu=0

# Dataset path
DATA_DIR=./dataset/TextZoom

# Model configuration
MODEL_FLAGS="\
--large_size 512 \
--small_size 128 \
--class_cond False \
--diffusion_steps 1000 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 128 \
--num_heads 1 \
--num_res_blocks 2 \
--resblock_updown True \
--attention_resolutions 16,8 \
--dropout 0.1 \
--use_fp16 False \
--use_scale_shift_norm True \
--use_checkpoint False"

# Training configuration
TRAIN_FLAGS="\
--kb 1 \
--use_gt_dimss 1 \
--use_synthesizer 0 \
--use_degrader 0 \
--gpu ${gpu} \
--lr 3e-4 \
--batch_size 32 \
--schedule_sampler uniform \
--timestep_respacing 100 \
--use_kl False \
--predict_xstart True \
--rescale_timesteps True \
--rescale_learned_sigmas False \
--log_interval 50 \
--save_interval 10000 \
--max_steps 150000 \
--data_dir ${DATA_DIR}"

# Run training
python train_on_textzoom.py $TRAIN_FLAGS $MODEL_FLAGS

"""
This script launches training for the TextZoom dataset, setting up GPU, model, and training hyperparameters, 
then runs train_on_textzoom.py with those configurations to train a diffusion-based image super-resolution model.
"""