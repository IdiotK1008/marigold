#!/usr/bin/env bash
set -e
set -x

bash script/eval/71_infer_sws.sh
bash script/eval/72_eval_sws.sh