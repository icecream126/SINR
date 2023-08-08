import glob
import numpy as np

filenames = glob.glob('./dataset/era5'+'/*.npz')
filenames = sorted(filenames)
unit = 1/len(filenames)

for cnt, filename in enumerate(filenames):
    x = dict(np.load(filename))
    x['time'] = np.array(cnt*unit)
    np.savez(filename, **x)