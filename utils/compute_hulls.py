import h5py
import argparse
import pickle
from scipy.spatial import Delaunay
from imagenet_utils import *
from dbscan import dbscan_filter
from tqdm import tqdm

def compute_hulls(model_path, labels, dbscan=False, reduced=False):
    hulls = {}
    for label in tqdm(labels, total=len(labels)):
        encodings = get_concept_encodings(label, model_path, 'train', reduced=reduced)
        if dbscan:
            encodings = dbscan_filter(encodings)
        d = Delaunay(encodings)
        hulls[label] = d
    hulls_path = model_path + '/hulls'
    if reduced:
        hulls_path += '_2d'
    hulls_path += '.p'
    with open(hulls_path, 'wb+') as f:
        pickle.dump(hulls, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--dbscan', action='store_true')
    args = parser.parse_args()

    labels = []
    with open('/data/nlp/bless/wbless_imagenet_labels.txt') as f:
        labels += f.read().splitlines()
    with open('/data/nlp/hyperlex/hyperlex_imagenet_labels.txt') as f:
        labels += f.read().splitlines()
    labels += list(train_indices.keys())
    labels = list(set(labels))

    model_path = 'model/vae' + str(args.latent_size)
    compute_hulls(model_path, labels, args.dbscan, args.reduced)
