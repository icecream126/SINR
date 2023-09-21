# CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
#     --dataset_dir dataset/spatial/era5_geopotential_100 \
#     --model ginr \
#     --max_epochs 100 \
#     --batch_size 1 \
#     --skip

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/temporal_1_00/era5_geopotential \
    --model ginr \
    --max_epochs 100 \
    --n_fourier 34 \
    --levels 6 \
    --batch_size 4096 \
    --lr 0.0001 \
    --skip \
    --normalize