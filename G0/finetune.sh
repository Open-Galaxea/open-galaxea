#!/bin/bash
# This script is used to finetune the model
# the first argument is the gpu number
GPU=$1
config=$2
ARGS=${@:3}

torchrun --standalone --nnodes 1 --nproc-per-node $GPU finetune.py --config $config $ARGS
