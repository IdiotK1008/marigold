#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-v1-0"}
subfolder=${2:-"eval"}

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 30 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_sws_eval.yaml \
    --output_dir output/${subfolder}/sws_eval/prediction \
