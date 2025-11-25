#!/bin/bash

PROJECT_ROOT=./

CUDA_VISIBLE_DEVICES=0 python run.py \
paths.root_dir=$PROJECT_ROOT \
experiment=main/cada/cada-50.yaml \
env=leaveout-id.yaml