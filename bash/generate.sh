# ERA5
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

### Spatial ###
# Resolution 0.25
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_temperature
python src/datasets/generation/era5.py --variable=sea_surface_temperature --mode=single --level_type=single --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_sea_temperature
python src/datasets/generation/era5.py --variable=surface_pressure --mode=single --level_type=single --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_precipitation
python src/datasets/generation/era5.py --variable=high_cloud_cover --mode=single --level_type=single --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_high_cloud
python src/datasets/generation/era5.py --variable=100m_u_component_of_wind --mode=single --level_type=single --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_100m_u_component_of_wind


# Resolution 0.50
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_temperature
python src/datasets/generation/era5.py --variable=v_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_v_wind
python src/datasets/generation/era5.py --variable=sea_surface_temperature --mode=single --level_type=single --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_sea_temperature
python src/datasets/generation/era5.py --variable=total_precipitation --mode=single --level_type=single --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_precipitation
python src/datasets/generation/era5.py --variable=high_cloud_cover --mode=single --level_type=single --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_high_cloud
python src/datasets/generation/era5.py --variable=100m_u_component_of_wind --mode=single --level_type=single --years=2000 --resolution=0.50 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_50/era5_100m_u_component_of_wind



# Resolution 1.00
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_cloud
python src/datasets/generation/era5.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_temperature
python src/datasets/generation/era5.py --variable=v_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_v_wind
python src/datasets/generation/era5.py --variable=sea_surface_temperature --mode=single --level_type=single --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_sea_temperature
python src/datasets/generation/era5.py --variable=total_precipitation --mode=single --level_type=single --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_precipitation
python src/datasets/generation/era5.py --variable=100m_u_component_of_wind --mode=single --level_type=single --years=2000 --resolution=1.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_1_00/era5_100m_u_component_of_wind




# Resolution 2.00
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=2.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_00/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=2.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_00/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=2.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_00/era5_cloud


# Resolution 4.00
python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=4.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_4_00/era5_geopotential
python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=4.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_4_00/era5_wind
python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=4.00 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_4_00/era5_cloud



# # Resolution 2.8125
# python src/datasets/generation/era5.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=2.8125 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_geopotential
# python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=single --level_type=pressure --years=2000 --resolution=2.8125 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_wind
# python src/datasets/generation/era5.py --variable=cloud_cover --mode=single --level_type=pressure --years=2000 --resolution=2.8125 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_2_8125/era5_cloud

# ### Temporal ###
# # Resolution 0.25
# python src/datasets/generation/era5.py --variable=geopotential --mode=separate --level_type=pressure --years=2000 --resolution=0.25 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_0_25/era5_geopotential
# python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=separate --level_type=pressure --years=2000 --resolution=0.25 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_0_25/era5_wind
# python src/datasets/generation/era5.py --variable=cloud_cover --mode=separate --level_type=pressure --years=2000 --resolution=0.25 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_0_25/era5_cloud

# # Resolution 1.00
# python src/datasets/generation/era5.py --variable=geopotential --mode=separate --level_type=pressure --years=2000 --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_1_00/era5_geopotential
# python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=separate --level_type=pressure --years=2000 --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_1_00/era5_wind
# python src/datasets/generation/era5.py --variable=cloud_cover --mode=separate --level_type=pressure --years=2000 --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_1_00/era5_cloud

# # Resolution 2.8125
# python src/datasets/generation/era5.py --variable=geopotential --mode=separate --level_type=pressure --years=2000 --resolution=2.8125 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_2_8125/era5_geopotential
# python src/datasets/generation/era5.py --variable=u_component_of_wind --mode=separate --level_type=pressure --years=2000 --resolution=2.8125 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_2_8125/era5_wind
# python src/datasets/generation/era5.py --variable=cloud_cover --mode=separate --level_type=pressure --years=2000 --resolution=2.8125 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal_2_8125/era5_cloud


# # spherical harmonics
# python src/datasets/generation/spherical_harmonics.py --l=3 --m=3 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics
# python src/datasets/generation/spherical_harmonics.py --l=5 --m=3 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics
# python src/datasets/generation/spherical_harmonics.py --l=7 --m=3 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics
# python src/datasets/generation/spherical_harmonics.py --l=20 --m=19 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics
# python src/datasets/generation/spherical_harmonics.py --l=30 --m=29 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics
# python src/datasets/generation/spherical_harmonics.py --l=40 --m=39 --res=1 --custom_fn=data.npz --output_dir ./dataset/spatial/spherical_harmonics