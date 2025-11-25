#!/bin/bash

PROJECT_ROOT=./

python test.py --model transformer --checkpoint "../../../checkpoint/RF-TE/id/50/1.ckpt","../../../checkpoint/RF-TE/id/50/2.ckpt","../../../checkpoint/RF-TE/id/50/3.ckpt"
