# Dataset
Run the following codes to download datasets.
```
# Resolution 0.25
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_temperature

# Resolution 0.50
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_temperature

# Resolution 1.00
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_temperature
```
# How to sweep
Yaml files for sweep is in ```sweeps/``` folder. 
## Meaning of subdirectories
```era5_spatial_sr_2x/``` : 2x spatial super resolution
```era5_spatial_sr_4x/``` : 4x spatial super resolution

TODO : Add GINR sweep



## How to run sweep
```task_directory : [era5_spatial_sr_2x, era5_spatial_sr_4x]```
```sweep_file : [gauss_3d.yaml, healpix.yaml, ngp_interp_2d.yaml, relu.yaml, shinr.yaml, siren.yaml, wire.yaml]```
```
# First create sweep agent
wandb sweep --project [YOUR PROJECT NAME] ./sweeps/[task_directory]/[sweep_file]

# Then sweep agent will be created. ex) wandb agent ~~

# Run a sweep 
CUDA_VISIBLE_DEVICES=0 wandb agent ~~
```


# How to run
## Spatial SR
### Arguments
* **dataset_dir** (str) ```[dataset/spatial_0_25/era5_geopotential, dataset/spatial_0_25/era5_temperature]```
* **downscale_factor** (int) ```[2,4]```
* **seed** (int) ```[0,1,2]```
* **model** (str) ```[healpix, gauss, relu, gauss, shinr, wire, siren]```

### Example script
```
# Model : HEALPix
# Dataset : Geopotential
# Downscale factor : x2
# Seed : 0

CUDA_VISIBLE_DEVICES=0 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_geopotential \
    --downscale_factor 2 \
    --seed 0 \
    --n_levels 9 \
    --n_features_per_level 2 \
    --input_dim 2 \
    --batch_size 4096 \
    --model healpix \
    --normalize \
    --skip \
    --plot
```

## Temporal SR
TBU

## Denoising
TBU

## Compression
TBU