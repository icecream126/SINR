# ERA5
# Spatial
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/spatial/era5_cloud

# Temporal
python src/datasets/generation/era5.py --variable=geopotential --mode=separate --level_type=pressure --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=separate --level_type=pressure --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=separate --level_type=pressure --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_cloud

