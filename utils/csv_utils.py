import h5py
import sys
import numpy as np
from tqdm import tqdm
from imagenet_utils import *
import random

def read_csv(path):
    data = []
    labels = []
    with open(path) as f:
        for line in f:
            entry = line.rstrip().split(',')
            data.append([float(s) for s in entry[:-1]])
            labels.append(int(entry[-1]))
    data = np.array(data)
    return data, labels

def write_csv(data, labels, path):
    with open(path, 'w+') as f:
        for enc, label in zip(data, labels):
            f.write(','.join([str(i) for i in enc] + [str(label)]) + '\n')

def subsample(input, output, k):
    lines = []
    with open(input) as f:
        for line in f:
            lines.append(line.rstrip())
    sample = random.sample(lines, k)
    with open(output, 'w+') as f:
        f.write('\n'.join(sample))

def extract_csv(label, model, subsample=True):
    encodings, filenames = get_concept_encodings(label, model, 'train', include_filenames=True)
    if len(encodings) > 10000:
        indices = list(np.arange(len(encodings)))
        sample = random.sample(indices, 10000)
        encodings, filenames = encodings[sample], filenames[sample]
    labels = [os.path.split(filename)[0].decode('utf-8') for filename in filenames]
    unique_labels = list(set(labels))
    label_indices = [unique_labels.index(label) for label in labels]
    write_csv(encodings, label_indices, model + '/csv/' + label)
    with open(os.path.join(model, 'csv', label + '_filenames'), 'w+') as f:
        f.write('\n'.join([s.decode('utf-8') for s in filenames]))

def read_clusters(path):
    clusters = []
    with open(path) as f:
        for line in f:
            clusters.append(int(line.rstrip()))
    clusters = np.array(clusters)
    return clusters

def read_filenames(path):
    with open(path) as f:
        return f.read().splitlines()

def get_cluster_encodings(model, label, include_outliers):
    encodings, _ = read_csv(os.path.join(model, 'csv', label))
    clusters = read_clusters(os.path.join(model, 'csv', label + '_clusters'))
    data = []
    for (enc, cluster) in zip(encodings, clusters):
        if include_outliers or not cluster == -1:
            data.append(enc)
    return np.array(data)

if __name__ == '__main__':
    for label in get_all_labels():
        extract_csv(label, 'model/vae2')
