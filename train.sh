# NOAA

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset noaa \
    --model relu \
    --dataset_dir dataset/noaa/tcdcclm \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset noaa \
    --model siren \
    --dataset_dir dataset/noaa/tcdcclm \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset noaa \
    --model shinr \
    --dataset_dir dataset/noaa/tcdcclm \
    --batch_size 4 \
    --spherical \
    --time \
    --in_memory \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --dataset noaa \
    --model swinr \
    --dataset_dir dataset/noaa/tcdcclm \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset noaa \
    --model wire \
    --dataset_dir dataset/noaa/tcdcclm \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epoch 1000

# ERA5

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset era5 \
    --model relu \
    --dataset_dir dataset/era5 \
    --time \
    --max_epoch 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset era5 \
    --model siren \
    --dataset_dir dataset/era5 \
    --time \
    --max_epoch 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --dataset era5 \
    --model shinr \
    --dataset_dir dataset/era5 \
    --spherical \
    --time \
    --max_epoch 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --dataset era5 \
    --model swinr \
    --dataset_dir dataset/era5 \
    --time \
    --max_epoch 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset era5 \
    --model wire \
    --dataset_dir dataset/era5 \
    --time \
    --max_epoch 100 \
    --temporal_res 6 \
    --spatial_res 8


# CIRCLE

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset circle \
    --model relu \
    --dataset_dir dataset/circle \
    --batch_size 256 \
    --hidden_dim 32 \
    --max_epoch 100