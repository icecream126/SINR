# ERA5

python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial/era5_geopotential
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/spatial/era5_temperature

python src/datasets/generation/era5.py --variable=geopotential --mode=separate --level_type=pressure --resolution=2.8125 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_geopotential
python src/datasets/generation/era5.py --variable=temperature --mode=separate --level_type=pressure --resolution=2.8125 --pressure_level=850 --custom_fn=data.nc --output_dir=dataset/temporal/era5_temperature

# CIRCLE

python src/datasets/generation/circle.py --res=0.25 --radius=45 --custom_fn=data.npz --output_dir=dataset/circle