#!/usr/bin/env bash
set -e
set -x

bash script/eval/63_infer_kitti360_abs.sh
bash script/eval/64_eval_kitti360_abs.sh