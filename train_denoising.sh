# CIRCLE (Need search)

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/circle \
    --model swinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --sigma 20 \
    --hidden_dim 128 \
    --lr 0.007

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/circle \
    --model shinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --levels 2\
    --hidden_dim 128 \
    --lr 0.003

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/circle \
    --model relu \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 64 \
    --lr 0.005

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/circle \
    --model siren \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 0.1 \
    --hidden_dim 256 \
    --lr 0.008

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/circle \
    --model wire \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 0.001 \
    --sigma 1 \
    --hidden_dim 512 \
    --lr 0.01

# SUN360 (Need search)

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/sun360 \
    --model swinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 30 \
    --sigma 1 \
    --hidden_dim 512 \
    --lr 0.0002

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/sun360 \
    --model shinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --levels 4\
    --hidden_dim 512 \
    --lr 0.0003

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/sun360 \
    --model relu \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 512 \
    --lr 0.0001

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/sun360 \
    --model siren \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --hidden_dim 256 \
    --lr 0.0004

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/sun360 \
    --model wire \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --sigma 20 \
    --hidden_dim 128 \
    --lr 0.001

# Flickr360 

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/flickr360 \
    --model swinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 30 \
    --sigma 1 \
    --hidden_dim 512 \
    --lr 0.0004

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/flickr360 \
    --model shinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --levels 4\
    --hidden_dim 512 \
    --lr 0.0001

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/flickr360 \
    --model relu \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 128 \
    --lr 0.004

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/flickr360 \
    --model siren \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 0.1 \
    --hidden_dim 128 \
    --lr 0.005

CUDA_VISIBLE_DEVICES=0 python src/main_denoising.py \
    --dataset_dir dataset/flickr360 \
    --model wire \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --sigma 1 \
    --hidden_dim 64 \
    --lr 0.003
