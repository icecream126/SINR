This is code repository for SINR.

You can train the model with following commands : 

Dataset : dpt2m
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ginr.py --dataset_dir dataset/weather_time_dpt2m_cut/   --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
```

Dataset : tcdcclm
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ginr.py --dataset_dir dataset/weather_time_tcdcclm_cut/   --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
```

Dataset : gustsfc
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ginr.py --dataset_dir dataset/weather_time_gustsfc_cut/   --n_fourier 3 --n_nodes_in_sample 5000 --lr 0.001 --n_layers 8 --skip=True --time=True
```