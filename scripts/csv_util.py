import pandas as pd
import h5py
import os
from datasets import eval_labels
from builder import mmid, imagenet
import numpy as np
from tqdm import tqdm
import random

def extract():
    with h5py.File('model/combined/train_encodings.h5') as f:
        encodings = np.array(f['encodings'])
    with h5py.File('data/combined/train.h5') as f:
        labels = np.array(f['labels'])
        filenames = np.array(f['filenames'])
    label_set = [s.decode('utf-8') for s in list(set(labels))]

    indices = {}
    idx = 0
    n = 0
    c = None
    for i, l in tqdm(enumerate(labels), total=len(labels)):
        l = l.decode('utf-8')
        if not l == c:
            if n > 0:
                indices[l] = (idx, idx + n)
                n = 0
            idx = i
            c = l
        n += 1
    indices[l] = (idx, idx + n)

    for label in tqdm(eval_labels, total=len(eval_labels)):
        s = []
        if label in mmid:
            s += mmid[label]
        if label in imagenet:
            s += imagenet[label]
        enc = []
        fnames = []
        label_indices = []
        for l in s:
            l = l.replace('/', '_')
            if l not in indices:
                continue
            i, j = indices[l]
            enc += list(encodings[i:j])
            fnames += list(filenames[i:j])
            label_indices += [label_set.index(l)] * (j - i)
        if len(enc) > 10000:
            idx = np.array(random.sample(range(len(enc)), 10000))
            enc = np.array(enc)[idx].tolist()
            fnames = np.array(fnames)[idx].tolist()
            label_indices = np.array(label_indices)[idx].tolist()
        with open(os.path.join('model/combined/csv/train_encodings', label), 'w+') as f:
            for i in range(len(enc)):
                e = enc[i]
                li = label_indices[i]
                f.write(','.join([str(j) for j in e]) + ',' + str(li) + '\n')
        with open(os.path.join('model/combined/csv/train_filenames', label), 'w+') as f:
            f.write('\n'.join([j.decode('utf-8') for j in fnames]))

if __name__ == '__main__':
    extract()
