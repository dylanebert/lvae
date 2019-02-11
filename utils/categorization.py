from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.spatial import Delaunay
from tqdm import tqdm
import re

def classical(model_path, results_path, reduced):
    hulls_path = model_path + '/hulls'
    if reduced:
        hulls_path += '_2d'
    hulls_path += '.p'
    with open(hulls_path, 'rb') as f:
        hulls = pickle.load(f)
    with open(results_path, 'w+') as f:
        for label in tqdm(train_indices.keys(), total=len(train_indices)):
            hull = hulls[label]
            encodings, false_encodings = get_test_encodings(label, model_path, reduced=reduced)
            for enc in encodings:
                if hull.find_simplex(enc) >= 0:
                    f.write('{0}\t{1}\t{2}\n'.format(label, 1, 1))
                else:
                    f.write('{0}\t{1}\t{2}\n'.format(label, 0, 1))
            for enc in false_encodings:
                if hull.find_simplex(enc) >= 0:
                    f.write('{0}\t{1}\t{2}\n'.format(label, 0, 1))
                else:
                    f.write('{0}\t{1}\t{2}\n'.format(label, 0, 0))

def prototype(model_path, results_path, reduced):
    normals_path = model_path + '/normals'
    if reduced:
        normals_path += '_2d'
    normals_path += '.p'
    path = model_path + '/prototypes_test'
    if reduced:
        path += '_2d'
    path += '.p'
    with open(path, 'rb') as f:
        prototypes = pickle.load(f)
    with open(normals_path, 'rb') as f:
        normals = pickle.load(f)
    with open(results_path, 'w+') as f:
        for label in tqdm(train_indices.keys(), total=len(train_indices)):
            normal = normals[label]
            normalizer = normal.pdf(normal.mean)
            encodings, false_encodings = get_test_encodings(label, model_path, reduced=reduced)
            for enc in encodings:
                f.write('{0}\t{1}\t{2}\n'.format(label, 1, normal.pdf(enc) / normalizer))
            for enc in false_encodings:
                f.write('{0}\t{1}\t{2}\n'.format(label, 0, normal.pdf(enc) / normalizer))

def categorization(latent_size, method, reduced):
    model_path = 'model/vae' + str(latent_size)
    results_path = 'results/' + str(latent_size) + '/categorization/' + method
    if reduced:
        results_path += '_2d'
    results_path += '.txt'
    if method == 'classical':
        classical(model_path, results_path, reduced)
    elif method == 'prototype':
        prototype(model_path, results_path, reduced)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='classical or prototype', type=str, required=True)
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()

    categorization(args.latent_size, args.method, args.reduced)
