import h5py
from imagenet_utils import *
import os
from tqdm import tqdm

with h5py.File('model/vae2/encodings.hdf5') as f:
    encodings = f['encodings']
    for key, (i, n) in tqdm(train_indices.items(), total=len(train_indices)):
        with open(os.path.join('/home/dylan/Documents/cgal/data', key), 'w+') as g:
            for enc in encodings[i:i+n]:
                g.write(' '.join([str(s) for s in enc]) + '\n')
