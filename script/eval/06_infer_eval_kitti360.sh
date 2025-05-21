#!/usr/bin/env bash
set -e
set -x

bash script/eval/61_infer_kitti360.sh
bash script/eval/62_eval_kitti360.sh