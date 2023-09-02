# ERA5 Temporal

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/temporal/era5_geopotential \
    --model swinr \
    --max_epochs 100 \
    --batch_size 1 \
    --skip 

# ERA5 Spatial

CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset_dir dataset/spatial/era5_temperature \
    --model swinr \
    --skip \
    --plot


# CIRCLE

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/circle \
    --model shiren \
    --max_epochs 100 \
    --skip 

# SUN360

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/sun360 \
    --model shiren \
    --skip \
    --normalize \
    --max_epochs 10 \
    --plot

# Flickr360

CUDA_VISIBLE_DEVICES=0 python src/main_denoise.py \
    --dataset_dir dataset/flickr360 \
    --model swinr \
    --skip \
    --normalize \
    --max_epochs 10 \
    --plot
    --panorama_idx 0

