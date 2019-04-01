import h5py
import numpy as np
from datasets import eval_labels

def read_encodings(label, type):
    with h5py.File('model/combined/' + type + '_encodings.h5') as f:
        encodings = np.array(f['encodings'])
    with h5py.File('data/combined/' + type + '.h5') as f:
        filenames = [s.decode('utf-8') for s in f['filenames']]


read_encodings('test', 'test')
