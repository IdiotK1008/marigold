#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --alignment fixed_log \
    --prediction_dir output/${subfolder}/kitti_eigen_test/prediction \
    --output_dir output/${subfolder}/kitti_eigen_test/eval_metric \
