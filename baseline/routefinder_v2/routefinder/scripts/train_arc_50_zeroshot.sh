#!/bin/bash

PROJECT_ROOT=./

CUDA_VISIBLE_DEVICES=0 python run.py \
paths.root_dir=$PROJECT_ROOT \
experiment=main/rf/arc-50.yaml \
env=leaveout-zeroshot.yaml