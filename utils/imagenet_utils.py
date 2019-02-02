import os
import pickle
import h5py
import numpy as np
import sys
from wordnet_utils import *
import random
from collections import defaultdict

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

def get_exclusive_encodings(concepts, model_path, stacked=True):
    encodings_dict = {}
    for concept in concepts:
        encodings_dict[concept] = get_concept_encodings(concept, model_path, stacked=False)
    labels_dict = defaultdict(list)
    for concept in concepts:
        for label in encodings_dict[concept].keys():
            labels_dict[label].append(concept)
    for label in labels_dict.keys():
        c = labels_dict[label]
        k = len(c)
        if k <= 1:
            continue
        encodings = encodings_dict[c[0]][label]
        encodings = np.array_split(encodings, k)
        i = 0
        for c in c:
            encodings_dict[concept][label] = encodings[i]
            i += 1
    if stacked:
        return [np.concatenate(list(encodings_dict[concept].values())) for concept in concepts]
    else:
        return [encodings_dict[concept] for concept in concepts]

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
