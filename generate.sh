# ERA5

### Spatial & Temporal ###
### Time : 2000 - 01 - 01 ###
# Resolution 0.25
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_cloud

# Resolution 1.00
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_cloud

# Resolution 2.8125
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_cloud