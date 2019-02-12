import h5py
import argparse
import pickle
from scipy.stats import gaussian_kde
from sklearn.cluster import MeanShift
from imagenet_utils import *
from dbscan import dbscan_filter
from tqdm import tqdm
import random

def compute_kdes(model_path, labels, dbscan=False, reduced=False, enc_type='train'):
    kdes = {}
    for label in tqdm(labels, total=len(labels)):
        encodings = get_concept_encodings(label, model_path, enc_type, reduced=reduced)
        if dbscan:
            encodings = dbscan_filter(encodings)
        if encodings.shape[0] > 1000:
            encodings = np.array(random.sample(list(encodings), 1000))
        kernel = gaussian_kde(encodings.T)
        ms = MeanShift().fit(encodings)
        cc = ms.cluster_centers_
        max = np.argmax([kernel(c) for c in cc])
        prototype = cc[max]
        kdes[label] = [kernel, prototype]
    kdes_path = model_path + '/kdes_' + enc_type
    if reduced:
        kdes_path += '_2d'
    kdes_path += '.p'
    with open(kdes_path, 'wb+') as f:
        pickle.dump(kdes, f)

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
    compute_kdes(model_path, labels, args.dbscan, args.reduced, 'train')
    compute_kdes(model_path, labels, args.dbscan, args.reduced, 'dev')
    compute_kdes(model_path, labels, args.dbscan, args.reduced, 'test')
