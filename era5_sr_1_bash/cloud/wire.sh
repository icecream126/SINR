#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model wire \
    --max_epochs 500 \
    --lr 0.001 \
    --hidden_dim 256 \
    --omega 1 \
    --sigma 1 \
    --normalize \
    --skip \
    --plot