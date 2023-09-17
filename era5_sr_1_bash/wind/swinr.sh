#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model relu \
    --lr 0.0001 \
    --hidden_dim 256 \
    --omega 50 \
    --sigma 1 \
    --normalize \
    --skip \
    --plot