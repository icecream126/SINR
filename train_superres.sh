
# HEALPIX - SUN - 2
CUDA_VISIBLE_DEVICES=1 python src/main_superres.py \
    --dataset_dir dataset/sun360/ \
    --panorama_idx 6 \
    --max_epochs 30 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot

# HEALPIX - WIND - 2x
CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 30 \
    --hidden_dim 512 \
    --hidden_layers 4 \
    --n_levels 9 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot

# HEALPIX - WIND - 4
CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 30 \
    --n_levels 4 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 1024 \
    --model healpix \
    --normalize \
    --skip \
    --plot


CUDA_VISIBLE_DEVICES=2 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --gauss_scale 2 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 1024 \
    --model gauss \
    --normalize \
    --skip \
    --plot

CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --gauss_scale 4 \
    --downscale_factor 2 \
    --input_dim 3 \
    --project_name 231120_ngp_2d3d \
    --batch_size 1024 \
    --model gauss \
    --normalize \
    --skip \
    --plot


CUDA_VISIBLE_DEVICES=4,5 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --panorama_idx 2 \
    --max_epochs 500 \
    --n_levels 10 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 4096 \
    --model healpix \
    --normalize \
    --skip \
    --plot


CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --panorama_idx 2 \
    --max_epochs 500 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 4096 \
    --model gauss \
    --gauss_scale 2 \
    --normalize \
    --skip \
    --plot

CUDA_VISIBLE_DEVICES=7 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --panorama_idx 2 \
    --max_epochs 500 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --input_dim 2 \
    --project_name 231120_ngp_2d3d \
    --batch_size 4096 \
    --model gauss \
    --gauss_scale 4 \
    --normalize \
    --skip \
    --plot

# NGP - WIND - 4
CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --n_levels 7 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --model ngp_interp \
    --normalize \
    --skip \
    --plot

# NGP - WIND - 2
CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_wind \
    --max_epochs 500 \
    --n_levels 7 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --model ngp_interp \
    --normalize \
    --skip \
    --plot

# NGP - SUN360 - 4
CUDA_VISIBLE_DEVICES=5 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --panorama_idx 0 \
    --max_epochs 500 \
    --n_levels 7 \
    --n_features_per_level 2 \
    --downscale_factor 4 \
    --model ngp_interp \
    --normalize \
    --skip \
    --plot

# NGP - SUN360 - 2
CUDA_VISIBLE_DEVICES=7 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --panorama_idx 0 \
    --max_epochs 500 \
    --n_levels 7 \
    --n_features_per_level 2 \
    --downscale_factor 2 \
    --model ngp_interp \
    --normalize \
    --skip \
    --plot




# Spherical harmonics
CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/spatial/spherical_harmonics_5_3 \
    --model swinr \
    --omega 1 \
    --sigma 10 \
    --max_epochs 1 \
    --skip \
    --plot 

CUDA_VISIBLE_DEVICES=1 python src/main_superres.py \
    --dataset_dir dataset/spatial/spherical_harmonics_5_3 \
    --model shinr \
    --levels 4 \
    --max_epochs 1 \
    --skip \
    --plot 

# Von mises
CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/spatial/von_mises_80_001 \
    --model swinr \
    --omega 1 \
    --sigma 10 \
    --max_epochs 1 \
    --skip \
    --plot 

# ERA5 Temporal

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/temporal/era5_geopotential \
    --model swinr \
    --max_epochs 1 \
    --batch_size 1 \
    --skip 

# ERA5 Spatial Geopotential
CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model ngp_interp \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0 \
    --mapping_size 3 \
    --task reg \
    --level 3

CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model learnable \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0 \
    --mapping_size 3 \
    --task reg \
    --level 5


CUDA_VISIBLE_DEVICES=1 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model gauss \
    --gauss_scale 2\
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0 \
    --task reg

CUDA_VISIBLE_DEVICES=2 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model swinr_learn_all \
    --omega_0 50.0 \
    --sigma_0 1.0 \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0 \
    --task sr

CUDA_VISIBLE_DEVICES=2 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model shinr \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0 \
    --task reg


CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model ewinr \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 4096 \
    --seed 0


CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial_1_00/era5_cloud \
    --model swinr_adap_all \
    --omega_0 10.0 \
    --sigma_0 1.0 \
    --lr 0.0001 \
    --skip \
    --plot \
    --normalize \
    --project_name re_era5_spatial \
    --hidden_dim 512 \
    --batch_size 128 \
    --seed 0

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model swinr \
    --omega 5 \
    --sigma 5 \
    --lr 0.003 \
    --skip \
    --plot

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_geopotential \
    --model relu \
    --omega 5 \
    --sigma 5 \
    --skip \
    --plot

# ERA5 Spatial Wind

CUDA_VISIBLE_DEVICES=7 python src/main_superres.py \
    --dataset_dir dataset/spatial/era5_wind \
    --model swinr \
    --omega 5 \
    --sigma 5 \
    --skip \
    --plot

# CIRCLE

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/circle \
    --model swinr \
    --max_epochs 500 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --sigma 20 \
    --hidden_dim 128 \
    --lr 0.007

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/circle \
    --model shinr \
    --max_epochs 500 \
    --skip \
    --plot \
    --normalize \
    --levels 4\
    --hidden_dim 128 \
    --lr 0.003

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/circle \
    --model relu \
    --max_epochs 500 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 64 \
    --lr 0.005

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/circle \
    --model siren \
    --max_epochs 500 \
    --skip \
    --plot \
    --normalize \
    --omega 0.1 \
    --hidden_dim 256 \
    --lr 0.008

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/circle \
    --model wire \
    --max_epochs 500 \
    --skip \
    --plot \
    --normalize \
    --omega 0.001 \
    --sigma 1 \
    --hidden_dim 512 \
    --lr 0.01

# SUN360


CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/sun360\
    --model swinr_pe \
    --lr 0.0003 \
    --skip \
    --plot \
    --normalize \
    --project_name re_spatial \
    --hidden_dim 512 \
    --batch_size 512 \
    --seed 0 \
    --panorama_idx 10

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/sun360\
    --model ewinr \
    --lr 0.0003 \
    --skip \
    --plot \
    --normalize \
    --project_name re_spatial \
    --hidden_dim 512 \
    --batch_size 512 \
    --seed 0 \
    --panorama_idx 10

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/sun360\
    --model gauss \
    --gauss_scale 2.0 \
    --lr 0.0003 \
    --skip \
    --plot \
    --normalize \
    --project_name re_spatial \
    --hidden_dim 512 \
    --batch_size 512 \
    --seed 0 \
    --panorama_idx 10

CUDA_VISIBLE_DEVICES=6 python src/main_superres.py \
    --dataset_dir dataset/sun360\
    --model swinr_learn_all \
    --omega_0 10.0 \
    --sigma_0 10.0 \
    --lr 0.0003 \
    --skip \
    --plot \
    --normalize \
    --project_name re_spatial \
    --hidden_dim 512 \
    --batch_size 512 \
    --seed 0 \
    --panorama_idx 10

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --model swinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 40 \
    --sigma 1 \
    --hidden_dim 512 \
    --lr 0.0003

CUDA_VISIBLE_DEVICES=1 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --model shinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --levels 4\
    --hidden_dim 512 \
    --lr 0.0003

CUDA_VISIBLE_DEVICES=2 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --model relu \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 512 \
    --lr 0.006

CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --model siren \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --hidden_dim 256 \
    --lr 0.002

CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/sun360 \
    --model wire \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --sigma 20 \
    --hidden_dim 128 \
    --lr 0.005

# Flickr360 (Need Search)

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/flickr360 \
    --model swinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 20 \
    --sigma 1 \
    --hidden_dim 64 \
    --lr 0.001

CUDA_VISIBLE_DEVICES=1 python src/main_superres.py \
    --dataset_dir dataset/flickr360 \
    --model shinr \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --levels 4\
    --hidden_dim 512 \
    --lr 0.0003

CUDA_VISIBLE_DEVICES=2 python src/main_superres.py \
    --dataset_dir dataset/flickr360 \
    --model relu \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --hidden_dim 128 \
    --lr 0.0001

CUDA_VISIBLE_DEVICES=3 python src/main_superres.py \
    --dataset_dir dataset/flickr360 \
    --model siren \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 1 \
    --hidden_dim 64 \
    --lr 0.002

CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/flickr360 \
    --model wire \
    --max_epochs 2000 \
    --skip \
    --plot \
    --normalize \
    --omega 0.01 \
    --sigma 10 \
    --hidden_dim 64 \
    --lr 0.002
