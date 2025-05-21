#!/usr/bin/env bash
set -e
set -x

python infer_latent.py  \
    --seed 1234 \
    --denoise_steps 30 \
    --ensemble_size 1 \
    --dataset_config dataset_eval_hypersim.yaml \
    --output_dir my_data/train/hypersim \

python infer_latent.py  \
    --seed 1234 \
    --denoise_steps 30 \
    --ensemble_size 1 \
    --dataset_config dataset_eval_vkitti.yaml \
    --output_dir my_data/train/vkitti \