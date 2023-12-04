# SUN360 - 2
CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/sun360/ \
    --panorama_idx 6 \
    --max_epochs 500 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --input_dim 2 \
    --project_name 231203_sr \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot

# SUN360 - 4
CUDA_VISIBLE_DEVICES=5 python src/main_superres.py \
    --dataset_dir dataset/sun360/ \
    --panorama_idx 6 \
    --max_epochs 500 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231203_sr \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot

# HEALPIX - WIND - 2
CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --input_dim 2 \
    --project_name 231203_sr \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot

# HEALPIX - WIND - 4
CUDA_VISIBLE_DEVICES=7 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231203_sr \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot