CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_path dataset/spatial/circle/data.npz \
    --model shinr \
    --max_epochs 1 \
    --skip \
    --plot