import pandas as pd
import h5py
import os
from datasets import eval_labels
from builder import mmid, imagenet
import numpy as np

def extract():
    with h5py.File('model/combined/train_encodings.h5') as f:
        encodings = np.array(f['encodings'])
    with h5py.File('data/combined/train.h5') as f:
        filenames = [s.decode('utf-8') for s in f['filenames']]
        labels = [s.decode('utf-8') for s in f['labels']]

    indices = {}
    idx = 0
    n = 0
    c = None
    for i, fname in enumerate(filenames):
        l = os.path.split(fname)[0]
        if not l == c:
            if n > 0:
                indices[l] = (idx, idx + n)
                n = 0
            idx = i
            c = l
        n += 1
    indices[l] = (idx, idx + n)

    for label in eval_labels:
        s = []
        if label in mmid:
            s += mmid[label]
        if label in imagenet:
            s += imagenet[label]
        with open(os.path.join('model/combined/csv', label), 'w+') as f:
            for l in s:
                l = l.replace('/', '_')
                if l in indices:
                    for i in range(indices[l][0], indices[l][1]):
                        f.write(','.join([str(j) for j in encodings[i]]) + '\n')

if __name__ == '__main__':
    extract()
