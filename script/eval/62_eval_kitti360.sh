#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_kitti360_eval.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/kitti360_eval/prediction \
    --output_dir output/${subfolder}/kitti360_eval/eval_metric \
