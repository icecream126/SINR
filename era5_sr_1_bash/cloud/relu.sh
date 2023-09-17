#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_geopotential \
    --model relu \
    --lr 0.0004 \
    --hidden_dim 256 \
    --max_epochs 500 \
    --normalize \
    --skip \
    --plot