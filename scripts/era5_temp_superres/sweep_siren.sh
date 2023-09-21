conda activate sinr
wandb sweep --project era5_temp_superres siren.yaml
# CUDA_VISIBLE_DEVICES=4 wandb agent postech_sinr/era5_temp_geo100/3q3q2q7l