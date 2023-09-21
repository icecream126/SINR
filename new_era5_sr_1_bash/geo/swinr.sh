#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model swinr \
    --max_epochs 500 \
    --lr 0.0003 \
    --hidden_dim 512 \
    --omega 20 \
    --sigma 1 \
    --normalize \
    --skip \
    --plot