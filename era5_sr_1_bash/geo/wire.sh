#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model relu \
    --max_epochs 500 \
    --lr 0.001 \
    --hidden_dim 128 \
    --omega 1 \
    --sigma 1 \
    --normalize \
    --skip \
    --plot