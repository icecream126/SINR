import numpy as np
import torch
import os

torch.manual_seed(1)
data_types = ['dpt2m', 'tcdcclm' , 'gustsfc']
n_nodes, train_size, valid_size = 65160, 39000, 13000

ids = torch.randperm(n_nodes)
train_idx, valid_idx, test_idx = ids[:train_size], ids[train_size:train_size+valid_size], ids[train_size+valid_size:]
prefixes = ('train', 'valid', 'test')

for data_type in data_types : 
    data_dir_path = './dataset/weather_time_'+data_type+'_cut/npz_files/'
    tmp_files = os.listdir(data_dir_path)
    data_files = [x for x in tmp_files if not x.startswith(prefixes)] # remove already generated train, valid data
    
    for data_file in data_files : 
        data = np.load(data_dir_path+data_file)
        spherical_points = np.load(data_dir_path[:-10]+'spherical_points.npy')
        time = data['time']
        train_target = data['target'][train_idx]
        train_spherical_points = spherical_points[train_idx]

        valid_target = data['target'][valid_idx]
        valid_spherical_points = spherical_points[valid_idx]

        test_target = data['target'][test_idx]
        test_spherical_points = spherical_points[test_idx]

        np.savez(data_dir_path+'train_'+data_file,time=time,target=train_target)
        np.savez(data_dir_path+'valid_'+data_file,time=time,target=valid_target)
        np.savez(data_dir_path+'test_'+data_file,time=time,target=test_target)
        np.save(data_dir_path[:-10]+'train_spherical_points.npy',train_spherical_points)
        np.save(data_dir_path[:-10]+'valid_spherical_points.npy',valid_spherical_points)
        np.save(data_dir_path[:-10]+'test_spherical_points.npy',test_spherical_points)
