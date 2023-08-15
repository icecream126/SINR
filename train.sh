# NOAA

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset tcdcclm \
    --model relu \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epochs 1000

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset tcdcclm \
    --model siren \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epochs 1000

CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --dataset tcdcclm \
    --model shinr \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epochs 1000

CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --dataset tcdcclm \
    --model swinr \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epochs 1000

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset tcdcclm \
    --model wire \
    --batch_size 4 \
    --time \
    --in_memory \
    --max_epochs 1000

# ERA5

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset era5_temp \
    --model relu \
    --max_epochs 1000

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --dataset era5 \
    --model siren \
    --time \
    --max_epochs 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --dataset era5 \
    --model shinr \
    --time \
    --max_epochs 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --dataset era5 \
    --model swinr \
    --time \
    --max_epochs 100 \
    --temporal_res 6 \
    --spatial_res 8

CUDA_VISIBLE_DEVICES=4 python src/main.py \
    --dataset era5 \
    --model wire \
    --time \
    --max_epochs 100 \
    --temporal_res 6 \
    --spatial_res 8

# CIRCLE

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --dataset circle \
    --model relu \
    --max_epochs 1

# SUN360

CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --dataset sun360 \
    --model relu \
    --plot \
    --max_epochs 100