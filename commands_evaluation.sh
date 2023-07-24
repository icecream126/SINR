# Weather modelling
python src/utils/plot/eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_gustsfc --n_fourier 100 --time=True --time_factor=2 --cmap Spectral --append gustsfc
python src/utils/plot/eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_dpt2m   --n_fourier 100 --time=True --time_factor=2 --cmap hot      --append dpt2m
python src/utils/plot/eval_weather.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_tcdcclm --n_fourier 100 --time=True --time_factor=2 --cmap Blues    --append tcdcclm

# Weather modelling + super-resolution (see comments in commands_training.sh)
python src/utils/plot/eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_gustsfc_sr_cut --n_fourier 34 --time=True --time_factor=2 --cmap Spectral --append gustsfc
python src/utils/plot/eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_dpt2m_sr_cut   --n_fourier 34 --time=True --time_factor=2 --cmap hot      --append dpt2m
python src/utils/plot/eval_weather_time_sr.py lightning_logs/.../checkpoints/best.ckpt --dataset_dir dataset/weather_time_tcdcclm_sr_cut --n_fourier 34 --time=True --time_factor=2 --cmap Blues    --append tcdcclm

