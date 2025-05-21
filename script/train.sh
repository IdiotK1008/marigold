#!/bin/bash
set -e
set -x
export CUDA_VISIBLE_DEVICES=2
python train.py \
    --config config/train/fisheye.yaml \
    --resume_run output/model/fisheye/checkpoint/latest \
    --output_dir output/model \
    --base_data_dir data \
    --base_ckpt_dir base_ckpt_dir/ \