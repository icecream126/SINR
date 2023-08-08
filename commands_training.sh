# NOAA

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset noaa \
    --model inr \
    --dataset_dir dataset/weather_time_dpt2m \
    --batch_size 8 \
    --time \
    --in_memory \
    --skip \
    --max_epoch 5000

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset noaa \
    --model inr \
    --dataset_dir dataset/weather_time_dpt2m \
    --batch_size 8 \
    --time \
    --in_memory \
    --skip \
    --sine \
    --all_sine \
    --max_epoch 5000

CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset noaa \
    --model shinr \
    --dataset_dir dataset/weather_time_dpt2m \
    --batch_size 8 \
    --spherical \
    --time \
    --in_memory \
    --skip \
    --max_epoch 5000

CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --dataset noaa \
    --model swinr \
    --dataset_dir dataset/weather_time_dpt2m \
    --batch_size 8 \
    --time \
    --in_memory \
    --skip \
    --max_epoch 5000

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset noaa \
    --model wire \
    --dataset_dir dataset/weather_time_dpt2m \
    --batch_size 8 \
    --hidden_dim 64 \
    --time \
    --in_memory \
    --skip \
    --max_epoch 5000

# ERA5

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset era5 \
    --model inr \
    --dataset_dir dataset/era5_temp2m_16x \
    --skip \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset era5 \
    --model inr \
    --dataset_dir dataset/era5_temp2m_16x \
    --skip \
    --sine \
    --all_sine \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset era5 \
    --model shinr \
    --dataset_dir dataset/era5_temp2m_16x \
    --spherical \
    --skip \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --dataset era5 \
    --model swinr \
    --dataset_dir dataset/era5_temp2m_16x \
    --skip \
    --max_epoch 1000

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset era5 \
    --model wire \
    --dataset_dir dataset/era5_temp2m_16x \
    --skip \
    --max_epoch 1000