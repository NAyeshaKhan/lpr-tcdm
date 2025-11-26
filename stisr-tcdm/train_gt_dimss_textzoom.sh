#!/bin/bash -e
#!/bin/bash -e

gpu=1

# -------- MODEL CONFIG (must match SR architecture) -------- #
#Originally --small_size 64 --large_size 128, but that's not giving good results. Retrain at 128x64
MODEL_FLAGS="\
--small_size 128 \
--large_size 64 \
--class_cond False \
--diffusion_steps 1000 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 128 \
--num_heads 1 \
--num_head_channels -1 \
--num_res_blocks 2 \
--resblock_updown True \
--use_fp16 False \
--use_scale_shift_norm True \
--attention_resolutions 16,8 \
"

# -------- TRAINING CONFIG -------- #
TRAIN_FLAGS="\
--kb 9 \
--use_gt_dimss 1 \
--use_synthesizer 0 \
--use_degrader 0 \
--gpu ${gpu} \
--lr 3e-4 \
--batch_size 32 \
--log_interval 50 \
--save_interval 10000 \
--max_steps 100000 \
--data_dir ./dataset/TextZoom \
"

# -------- RUN TRAINING -------- #
CUDA_VISIBLE_DEVICES=$gpu \
python train_on_textzoom.py \
    $TRAIN_FLAGS \
    $MODEL_FLAGS \
    --ckpt_dir ./ckpt/super_resolver
