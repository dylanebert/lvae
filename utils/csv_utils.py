import h5py
import sys
import numpy as np
from tqdm import tqdm
from imagenet_utils import *
import random

def hdf5_to_csv(input, output):
    with h5py.File(input) as f:
        encodings = np.array(f['encodings'])
    write_csv(encodings, output)

def csv_to_hdf5(input, output):
    data = read_csv(input)
    with h5py.File(input, 'w') as f:
        f.create_dataset('encodings', data=data)

def extract_csv(word, model):
    encodings, filenames = get_concept_encodings(get_label_from_word(word), model, 'train', include_filenames=True)
    labels = [os.path.split(filename)[0].decode('utf-8') for filename in filenames]
    unique_labels = list(set(labels))
    label_indices = [unique_labels.index(label) for label in labels]
    write_csv_with_labels(encodings, label_indices, model + '/csv/' + word)

def subsample(input, output, k):
    lines = []
    with open(input) as f:
        for line in f:
            lines.append(line.rstrip())
    sample = random.sample(lines, k)
    with open(output, 'w+') as f:
        f.write('\n'.join(sample))

def read_csv(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append([float(s) for s in line.rstrip().split(',')])
    data = np.array(data)
    return data

def read_csv_with_labels(path):
    data = []
    labels = []
    with open(path) as f:
        for line in f:
            entry = line.rstrip().split(',')
            data.append([float(s) for s in entry[:-1]])
            labels.append(int(entry[-1]))
    data = np.array(data)
    return data, labels

def write_csv(data, path):
    with open(path, 'w+') as f:
        for enc in tqdm(data, total=len(data)):
            f.write(','.join([str(i) for i in enc]) + '\n')

def write_csv_with_labels(data, labels, path):
    with open(path, 'w+') as f:
        for enc, label in tqdm(zip(data, labels), total=len(data)):
            f.write(','.join([str(i) for i in enc] + [str(label)]) + '\n')

def read_clusters(path):
    clusters = []
    with open(path) as f:
        for line in f:
            clusters.append(int(line.rstrip()))
    clusters = np.array(clusters)
    return clusters

if __name__ == '__main__':
    #extract_csv('bird_of_prey', 'model/vae3')
    #extract_csv('entity', 'model/vae2')
    subsample('model/vae2/csv/entity', 'model/vae2/csv/entity_sample', 10000)
