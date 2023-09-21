#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model shinr \
    --max_epochs 500 \
    --lr 0.0007 \
    --hidden_dim 256 \
    --levels 3 \
    --normalize \
    --skip \
    --plot