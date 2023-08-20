CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_path dataset/spatial/era5_geopotential/data.nc \
    --model shinr \
    --max_epochs 10 \
    --skip \
    --plot