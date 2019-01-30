import os
import pickle
import h5py
import numpy as np
import sys
from wordnet_utils import *
import random

with open('train_indices.p', 'rb') as f:
    train_indices = pickle.load(f)
with open('dev_indices.p', 'rb') as f:
    dev_indices = pickle.load(f)

def get_leaves(label):
    leaves = []
    labels = [label]
    i = 0
    while i < len(labels):
        label = labels[i]
        if label in train_indices:
            leaves.append(label)
        else:
            for hyponym in get_all_hyponyms(label):
                if hyponym in train_indices:
                    labels.append(hyponym)
        i += 1
    return leaves

def get_concept_embeddings(concept):
    leaves = get_leaves(concept)
    train_n = 0
    dev_n = 0
    for leaf in leaves:
        train_n += train_indices[leaf][1]
        dev_n += dev_indices[leaf][1]
    train = np.zeros((train_n, 2048))
    dev = np.zeros((dev_n, 2048))
    train_labels = {}
    dev_labels = {}
    idx = 0
    with h5py.File('train.hdf5') as f:
        for concept in leaves:
            i, n = train_indices[concept]
            train[idx:idx+n] = f['embeddings'][i:i+n]
            train_labels[concept] = (idx, n)
            idx += n
    idx = 0
    with h5py.File('dev.hdf5') as f:
        for concept in leaves:
            i, n = dev_indices[concept]
            dev[idx:idx+n] = f['embeddings'][i:i+n]
            dev_labels[concept] = (idx, n)
            idx += n
    return train, dev, train_labels, dev_labels

def get_concept_encodings(concept, model_path, stacked=True):
    leaves = get_leaves(concept)
    encodings = {}
    with h5py.File(os.path.join(model_path, 'encodings.hdf5')) as f:
        for concept in leaves:
            i, k = train_indices[concept]
            encodings[concept] = f['encodings'][i:i+k]
    if stacked:
        return np.concatenate(list(encodings.values()))
    else:
        return encodings

def get_random(n, k):
    data = np.zeros((n, k))
    i = 0
    with h5py.File('train.hdf5') as f:
        m = f['embeddings'].shape[0]
        while i < n:
            idx = random.randint(0, m-1)
            if i < n - 100:
                data[i:i+100] = f['embeddings'][idx:idx+100]
                i += 100
            else:
                data[i] = f['embeddings'][idx]
                i += 1
    return data
