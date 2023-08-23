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
    --model swinr \
    --batch_size 1 \
    --max_epochs 100 \
    --skip 

# SUN360

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_dir dataset/spatial/sun360 \
    --model swinr \
    --panorama_idx 0 \
    --skip \
    --omega 20 \
    --sigma 10 \
    --plot
