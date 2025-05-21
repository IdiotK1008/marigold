#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/my-model-v3-fisheye"}
subfolder=${2:-"eval"}
model=${3:-"resnet18"}

python infer_abs.py  \
    --checkpoint $ckpt \
    --dmvn_model $model \
    --dmvn_checkpoint base_ckpt_dir/dmvn_resnet/dmvnp_100.pth \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 30 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_kitti360_eval.yaml \
    --output_dir output/${subfolder}/kitti360/prediction \
