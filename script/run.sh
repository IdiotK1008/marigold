#!/usr/bin/env bash
set -e
set -x

CUDA_VISIBLE_DEVICES=1 \

python run.py \
    --checkpoint checkpoint/my-model-v0 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/vis\
    --output_dir output/vis/0

python run.py \
    --checkpoint checkpoint/my-model-v1 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/vis\
    --output_dir output/vis/1

python run.py \
    --checkpoint checkpoint/my-model-v2 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/vis\
    --output_dir output/vis/2

python run.py \
    --checkpoint checkpoint/my-model-v4 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/vis\
    --output_dir output/vis/4