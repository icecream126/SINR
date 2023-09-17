#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model relu \
    --max_epochs 500 \
    --lr 0.0007 \
    --hidden_dim 128 \
    --levels 5 \
    --normalize \
    --skip \
    --plot