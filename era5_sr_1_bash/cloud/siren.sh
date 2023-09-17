#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model relu \
    --lr 0.0005 \
    --hidden_dim 128 \
    --omega 1 \
    --normalize \
    --skip \
    --plot