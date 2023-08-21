CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset_path dataset/temporal/era5_geopotential \
    --model shinr \
    --max_epochs 1 \
    --skip \
    --batch_size 1 \
    # --plot