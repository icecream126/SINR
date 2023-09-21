conda activate sinr
wandb sweep --project era5_temp_superres shiren.yaml
# CUDA_VISIBLE_DEVICES=5 wandb agent postech_sinr/era5_temp_geo100/