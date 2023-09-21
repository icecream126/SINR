conda activate sinr
wandb sweep --project era5_temp_superres ginr.yaml
# CUDA_VISIBLE_DEVICES=5 wandb agent postech_sinr/era5_temp_geo100/3q3q2q7l