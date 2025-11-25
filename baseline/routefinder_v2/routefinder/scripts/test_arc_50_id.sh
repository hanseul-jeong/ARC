#!/bin/bash

PROJECT_ROOT=./

python test.py --model arc --checkpoint "../../../checkpoint/ARC/id/50/1.ckpt","../../../checkpoint/ARC/id/50/2.ckpt","../../../checkpoint/ARC/id/50/3.ckpt"
# paths.root_dir=$PROJECT_ROOT \
# experiment=main/rf/arc-50.yaml \
# env=leaveout-id.yaml