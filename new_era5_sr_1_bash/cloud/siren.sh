#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model siren \
    --max_epochs 500 \
    --lr 0.0005 \
    --hidden_dim 128 \
    --omega 1 \
    --normalize \
    --skip \
    --plot