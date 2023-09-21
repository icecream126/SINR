conda activate sinr
wandb sweep --project era5_temp_superres relu.yaml
# CUDA_VISIBLE_DEVICES=0 wandb agent postech_sinr/era5_temp_geo025/3q3q2q7l