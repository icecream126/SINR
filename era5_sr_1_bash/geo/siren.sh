#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model relu \
    --lr 0.0003 \
    --hidden_dim 64 \
    --omega 10 \
    --normalize \
    --skip \
    --plot