# ERA5 Temporal

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/temporal/era5_geopotential \
    --model swinr \
    --max_epochs 100 \
    --batch_size 1 \
    --skip 

# ERA5 Spatial

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model swinr \
    --max_epochs 100 \
    --skip 

# CIRCLE

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/circle \
    --model swinr \
    --batch_size 1 \
    --max_epochs 100 \
    --skip 

# SUN360

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/circle \
    --model swinr \
    --panorama_idx 0 \
    --max_epochs 100 \
    --skip 