import h5py
import argparse
import pickle
from scipy.stats import multivariate_normal
from imagenet_utils import *
from dbscan import dbscan_filter
from tqdm import tqdm

def compute_normals(model_path, labels, dbscan=False, reduced=False):
    normals = {}
    for label in tqdm(labels, total=len(labels)):
        encodings = get_concept_encodings(label, model_path, 'train', reduced=reduced)
        if dbscan:
            encodings = dbscan_filter(encodings)
        normals[label] = multivariate_normal(mean=np.mean(encodings, axis=0), cov=np.cov(encodings.T))
    normals_path = model_path + '/normals'
    if reduced:
        normals_path += '_2d'
    normals_path += '.p'
    with open(normals_path, 'wb+') as f:
        pickle.dump(normals, f)

def compute_prototypes(model_path, labels, dbscan=False, reduced=False, enc_type='dev'):
    prototypes = {}
    for label in tqdm(labels, total=len(labels)):
        encodings = get_concept_encodings(label, model_path, enc_type, reduced=reduced)
        if dbscan:
            encodings = dbscan_filter(encodings)
        prototypes[label] = np.mean(encodings, axis=0)
    prototypes_path = model_path + '/prototypes_' + enc_type
    if reduced:
        prototypes_path += '_2d'
    prototypes_path += '.p'
    with open(prototypes_path, 'wb+') as f:
        pickle.dump(prototypes, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--dbscan', action='store_true')
    args = parser.parse_args()

    labels = get_all_labels()
    model_path = 'model/vae' + str(args.latent_size)
    compute_normals(model_path, labels, args.dbscan, args.reduced)
    compute_prototypes(model_path, labels, args.dbscan, args.reduced, 'dev')
    compute_prototypes(model_path, labels, args.dbscan, args.reduced, 'test')
