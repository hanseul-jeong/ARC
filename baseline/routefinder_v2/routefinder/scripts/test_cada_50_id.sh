#!/bin/bash

PROJECT_ROOT=./

python test.py --model transformer --checkpoint "../../../checkpoint/CaDA/id/50/1.ckpt","../../../checkpoint/CaDA/id/50/2.ckpt","../../../checkpoint/CaDA/id/50/3.ckpt"
