CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --dataset era5 \
    --model inr \
    --dataset_dir dataset/era5_temp2m_16x \
    --batch_size 64 \
    --skip