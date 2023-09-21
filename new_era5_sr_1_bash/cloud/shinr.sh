#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model shinr \
    --lr 0.0007 \
    --hidden_dim 128 \
    --max_epochs 500 \
    --levels 5 \
    --normalize \
    --skip \
    --plot