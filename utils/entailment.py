from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.spatial import Delaunay
from tqdm import tqdm
import re

def classical(model_path, results_path, dset, enc_type, reduced):
    hulls_path = model_path + '/hulls'
    if reduced:
        hulls_path += '_2d'
    hulls_path += '.p'
    with open(hulls_path, 'rb') as f:
        hulls = pickle.load(f)
    with open(results_path, 'w+') as f:
        for (w1, w2, l1, l2, d) in tqdm(dset.pairs, total=len(dset.pairs)):
            hull = hulls[l2]
            w1_encodings = get_concept_encodings(l1, model_path, enc_type, reduced=reduced)
            n_entails = 0
            n = 0
            for enc in w1_encodings:
                if hull.find_simplex(enc) >= 0:
                    n_entails += 1
                n += 1
            p = n_entails / float(n)
            f.write('{0}\n'.format('\t'.join([str(i) for i in [w1, w2, l1, l2, d, p]])))

def prototype(model_path, results_path, dset, enc_type, reduced):
    normals_path = model_path + '/normals'
    if reduced:
        normals_path += '_2d'
    normals_path += '.p'
    with open(normals_path, 'rb') as f:
        normals = pickle.load(f)
    path = model_path + '/prototypes_' + enc_type
    if reduced:
        path += '_2d'
    path += '.p'
    with open(path, 'rb') as f:
        prototypes = pickle.load(f)
    with open(results_path, 'w+') as f:
        for (w1, w2, l1, l2, d) in tqdm(dset.pairs, total=len(dset.pairs)):
            normal = normals[l2]
            prototype = prototypes[l1]
            p = normal.pdf(prototype) / normal.pdf(normal.mean)
            f.write('{0}\n'.format('\t'.join([str(i) for i in [w1, w2, l1, l2, d, p]])))

def kde(model_path, results_path, dset, enc_type, reduced):
    kdes_path = model_path + '/kdes_train'
    if reduced:
        kdes_path += '_2d'
    kdes_path += '.p'
    with open(kdes_path, 'rb') as f:
        kdes = pickle.load(f)
    path = model_path + '/kdes_' + enc_type
    if reduced:
        path += '_2d'
    path += '.p'
    with open(path, 'rb') as f:
        prototypes = pickle.load(f)
    with open(results_path, 'w+') as f:
        for (w1, w2, l1, l2, d) in tqdm(dset.pairs, total=len(dset.pairs)):
            kde = kdes[l2]
            prototype = prototypes[l1][1]
            p = (kde[0](prototype) / kde[0](kde[1]))[0]
            f.write('{0}\n'.format('\t'.join([str(i) for i in [w1, w2, l1, l2, d, p]])))

def entailment(latent_size, enc_type, dset_name, method, reduced):
    model_path = 'model/vae' + str(latent_size)
    if dset_name == 'wbless':
        dset = WBless()
    elif dset_name == 'hyperlex':
        dset = Hyperlex()
    results_path = 'results/' + str(latent_size) + '/' + dset_name + '/' + method
    if reduced:
        results_path += '_2d'
    results_path += '_' + enc_type + '.txt'
    if method == 'classical':
        classical(model_path, results_path, dset, enc_type, reduced)
    elif method == 'prototype':
        prototype(model_path, results_path, dset, enc_type, reduced)
    elif method == 'kde':
        kde(model_path, results_path, dset, enc_type, reduced)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='classical or prototype', type=str, required=True)
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--dset', help='wbless or hyperlex', type=str, default='wbless')
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()

    entailment(args.latent_size, 'dev', args.dset, args.method, args.reduced)
    entailment(args.latent_size, 'test', args.dset, args.method, args.reduced)
