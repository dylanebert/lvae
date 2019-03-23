import os
import pickle
import h5py
import numpy as np
import sys
from wordnet_utils import *
import random
from collections import defaultdict

with open('/home/dylan/Documents/lvae/train_indices.p', 'rb') as f:
    train_indices = pickle.load(f)
with open('/home/dylan/Documents/lvae/dev_indices.p', 'rb') as f:
    dev_indices = pickle.load(f)
with open('/home/dylan/Documents/lvae/test_indices.p', 'rb') as f:
    test_indices = pickle.load(f)

def get_all_labels():
    labels = []
    with open('/data/nlp/bless/wbless_imagenet_labels.txt') as f:
        labels += f.read().splitlines()
    with open('/data/nlp/hyperlex/hyperlex_imagenet_labels.txt') as f:
        labels += f.read().splitlines()
    labels += list(train_indices.keys())
    labels = list(set(labels))
    return labels

def get_leaves(label):
    leaves = []
    labels = [label]
    seen = []
    i = 0
    while i < len(labels):
        label = labels[i]
        if label in seen:
            continue
        seen.append(label)
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

def get_concept_encodings(concept, model_path, encodings_type, stacked=True, reduced=False, include_filenames=False):
    leaves = get_leaves(concept)
    encodings = {}
    filenames = {}
    path = model_path + '/'
    if encodings_type == 'train':
        indices = train_indices
    elif encodings_type == 'dev':
        indices = dev_indices
        path += 'dev_'
    elif encodings_type == 'test':
        indices = test_indices
        path += 'test_'
    path += 'encodings'
    if reduced:
        path += '_2d'
    path += '.hdf5'
    with h5py.File(path) as f:
        with h5py.File(encodings_type + '.hdf5') as g:
            for concept in leaves:
                i, k = indices[concept]
                encodings[concept] = f['encodings'][i:i+k]
                filenames[concept] = g['filenames'][i:i+k]
    if stacked:
        if include_filenames:
            return np.concatenate(list(encodings.values())), np.concatenate(list(filenames.values()))
        else:
            return np.concatenate(list(encodings.values()))
    else:
        if include_filenames:
            return encodings, filenames
        else:
            return encodings

def get_exclusive_encodings(concepts, model_path, reduced=False):
    leaf_concepts = defaultdict(list)
    for concept in concepts:
        for leaf in get_leaves(concept):
            leaf_concepts[leaf].append(concept)
    encodings = defaultdict(list)
    for leaf, concepts_in_leaf in leaf_concepts.items():
        enc = get_concept_encodings(leaf, model_path, 'train', reduced=reduced)
        enc = np.array_split(enc, len(concepts_in_leaf))
        for i, concept in enumerate(concepts_in_leaf):
            encodings[concept].append(enc[i])
    return [np.concatenate(encodings[concept]) for concept in concepts]

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

def get_test_encodings(concept, model_path, reduced=False):
    labels = get_leaves(concept)
    encodings = []
    path = model_path + '/test_encodings'
    if reduced:
        path += '_2d'
    path += '.hdf5'
    with h5py.File(path) as f:
        set = np.arange(f['encodings'].shape[0])
        ex = []
        for label in labels:
            i, k = test_indices[label]
            encodings.append(list(f['encodings'][i:i+k]))
            ex.append(list(np.arange(i, i+k, dtype=int)))
        encodings = np.reshape(np.array(encodings), (-1, f['encodings'].shape[1]))
        ex = np.reshape(np.array(ex), (-1, 1))
        diff = np.setdiff1d(set, ex)
        sample_indices = sorted(random.sample(list(diff), encodings.shape[0]))
        enc = np.array(f['encodings'])
        false_encodings = enc[sample_indices]
        return encodings, false_encodings
