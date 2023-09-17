#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model relu \
    --lr 0.0007 \
    --hidden_dim 128 \
    --levels 5 \
    --normalize \
    --skip \
    --plot