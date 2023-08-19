CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset_path dataset/spatial/sun360/0.jpg \
    --model swinr \
    --max_epochs 100 \
    --skip \
    --plot