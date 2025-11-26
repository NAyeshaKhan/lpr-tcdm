#!/bin/bash -e

MODEL_FLAGS="--large_size 64 --small_size 128  --class_cond False --diffusion_steps 1000 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 1 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 40 --timestep_respacing 250"
model_path="./ckpt/super_resolver/ema_0.9999_100000.pt"
gpu=1

base_samples="./diff_samples/RLPR_preprocessed/RLPR_LR.npz"
out_path="--out_path ./diff_samples/hr_samples/RLPR_HR_generated.npz"
num_samples=200

args="--kb 9 --use_gt_dimss 1 --use_synthesizer 0 --gpu ${gpu} --model_path ${model_path} --num_samples ${num_samples} --base_samples ${base_samples} ${MODEL_FLAGS} ${SAMPLE_FLAGS} ${out_path}"

python super_resolver_degrader_sample.py $args
