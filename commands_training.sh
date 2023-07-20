# Weather modelling
# The first 66 eigenvectors are "useless" because the equiangular grid returned by
# GFS is non-uniform on the surface and has higher point density at the poles.
# The effect on the eigenvectors is that the low frequency ones are zero everywhere
# except the poles, and they tend to change a lot when sampling more points.
# To improve stability, we removed the first 66 eigenvectors manually and only
# train on the remaning 34.
# This is basically equivalent to training on the low-frequency eigenvectors
# as we did for the transferability experiments. The setting for the INR is also
# the same (ReLU, 8 layers, higher LR).
CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_gustsfc_cut/   --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_dpt2m_cut/   --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_tcdcclm_cut/ --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True

# set n_fourier == max_order of spherical harmonics