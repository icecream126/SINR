# Basic INRs
# python train_ginr.py --dataset_dir dataset/bunny_v1       --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
# python train_ginr.py --dataset_dir dataset/protein_1AA7_A --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True
# python train_ginr.py --dataset_dir dataset/us_elections   --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --skip=True --sine=True --all_sine=True

# Transferability: SBM
# Set max_epochs to 1000 because otherwise it takes too long and the performance
# python train_sbm.py --n_fourier 3 --max_epochs 1000

# Transferability: super-resolution
# Changes to improve transferability:
#   - Use ReLU instead of sine
#   - Add a couple of extra layers (because of ReLU)
#   - Lower n_fourier to 7 (found empirically)
#   - Increase learning rate to 0.001
# python train_ginr.py --dataset_dir dataset/bunny_v1/ --n_fourier 7 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True

# Conditional INR: reaction-diffusion
# python train_ginr.py --dataset_dir dataset/bunny_time/ --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --sine=True --all_sine=True --skip=True --time=True

# Conditional INR: multi-protein
# Use the --cut flag to control the size of the dataset
# srun python train_ginr.py --dataset_dir dataset/proteins --n_fourier 100 --n_nodes_in_sample 5000 --lr 0.0001 --n_layers 6 --latents --latent_dim=8 --sine=True --all_sine=True --skip=True --cut 100

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

for hidden_dim in 512 256 ; do 
    for lr in 0.0001 0.0005 0.001 0.005; do
        for latent_dim in 512 256 128 64 32; do
            for n_layers in 7 8 9; do
                for n_fourier in 2 3 4; do
                    CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_gustsfc_cut/ --n_fourier=${n_fourier} --n_nodes_in_sample 5000 --lr=${lr} --n_layers=${n_layers} --latent_dim=${latent_dim} --hidden_dim=${hidden_dim} --skip=True --time=True
                done
            done
        done
    done
done
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_dpt2m_cut/   --n_fourier 5 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
# CUDA_VISIBLE_DEVICES=2,3,5,6,7 python train_ginr.py --dataset_dir dataset/weather_time_tcdcclm_cut/ --n_fourier 5 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True

# set n_fourier == max_order of spherical harmonics