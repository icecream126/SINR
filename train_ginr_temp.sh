# CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
#     --dataset_dir dataset/spatial/era5_geopotential_100 \
#     --model ginr \
#     --max_epochs 100 \
#     --batch_size 1 \
#     --skip

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/temporal/era5_geopotential \
    --model ginr \
    --max_epochs 100 \
    --n_fourier 34 \
    --levels 8 \
    --batch_size 25600 \
    --lr 0.01 \
    --skip \
    --zscore_normalize