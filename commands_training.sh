CUDA_VISIBLE_DEVICES=0 python train_euclidean_inr.py \
    --dataset_dir dataset/weather_time_dpt2m_cut/ \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True

CUDA_VISIBLE_DEVICES=0 python train_spherical_inr.py \
    --dataset_dir dataset/weather_time_dpt2m_cut/ \
    --n_fourier 3 \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True

CUDA_VISIBLE_DEVICES=2 python train_graph_inr.py \
    --dataset_dir dataset/weather_time_dpt2m_cut/ \
    --n_fourier 34 \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True

CUDA_VISIBLE_DEVICES=3 python train_spherical_inr.py \
    --dataset_dir dataset/weather_time_gustsfc_cut/  \
     --n_fourier 3 \
     --n_nodes_in_sample 5000 \
     --lr 0.001 \
     --n_layers 8 \
     --skip=True \
     --time=True

CUDA_VISIBLE_DEVICES=4 python train_euclidean_inr.py \
    --dataset_dir dataset/weather_time_gustsfc_cut/  \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True

CUDA_VISIBLE_DEVICES=5 python train_graph_inr.py \
    --dataset_dir dataset/weather_time_gustsfc_cut/ \
    --n_fourier 34 \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True


CUDA_VISIBLE_DEVICES=6 python train_spherical_inr.py \
    --dataset_dir dataset/weather_time_tcdcclm_cut/ \
    --n_fourier 3 \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True

CUDA_VISIBLE_DEVICES=7 python train_euclidean_inr.py \
    --dataset_dir dataset/weather_time_tcdcclm_cut/ \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True


CUDA_VISIBLE_DEVICES=8 python train_graph_inr.py \
    --dataset_dir dataset/weather_time_tcdcclm_cut/ \
    --n_fourier 34 \
    --n_nodes_in_sample 5000 \
    --lr 0.001 \
    --n_layers 8 \
    --skip=True \
    --time=True
