import argparse
from wordnet_utils import *
from imagenet_utils import *
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy.stats import multivariate_normal, gaussian_kde
from dbscan import *

class Categorization():
    def __init__(self, model_path):
        self.model_path = model_path

    def classical(self, train, x, y_true):
        y_pred = []
        d = Delaunay(train)
        for enc, t in zip(x, y_true):
            if d.find_simplex(enc) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return np.array(y_pred)

    def kde(self, train, x):
        kernel = gaussian_kde(train.T)
        y_pred = []
        for enc in x:
            y_pred.append(kernel(enc)[0])
        return y_pred

    def prototype(self, train, x):
        kernel = multivariate_normal(mean=np.mean(train, axis=0), cov=np.cov(train.T))
        y_pred = []
        for enc in x:
            y_pred.append(kernel.pdf(enc))
        return y_pred

    def eval(self, method, save_path, dbscan=False):
        y_true_all = []
        y_pred_all = []
        for label in tqdm(train_indices.keys(), total=len(list(train_indices.keys()))):
            train = get_concept_encodings(label, os.path.join(self.model_path, 'encodings.hdf5'))
            if dbscan:
                train = dbscan_filter(train)
            x_true, x_false = get_test_encodings(label, self.model_path)
            x = np.concatenate([x_true, x_false])
            y_true = np.zeros((x_true.shape[0] + x_false.shape[0],))
            y_true[:x_true.shape[0]] = 1
            if method == 'classical':
                y_pred = self.classical(train, x, y_true)
            elif method == 'prototype':
                y_pred = self.prototype(train, x)
            else:
                y_pred = self.kde(train, x)
            y_true_all += list(y_true)
            y_pred_all += list(y_pred)
        with open(save_path, 'w+') as f:
            for y_true, y_pred in zip(y_true_all, y_pred_all):
                f.write('{0}\t{1}\n'.format(y_true, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--method', help='override method to only classical, prototype, kde (default classical and prototype)', type=str, default='')
    args = parser.parse_args()

    args.model_path = 'model/vae' + str(args.latent_size)
    args.save_dir = 'results/' + str(args.latent_size)

    model = Categorization(args.model_path)
    if args.method == 'classical' or args.method == '':
        if args.dbscan:
            filename = 'CLASSICAL_DBSCAN.txt'
        else:
            filename = 'CLASSICAL.txt'
        model.eval('classical', os.path.join(args.save_dir, 'categorization', filename), args.dbscan)
    if args.method == 'prototype' or args.method == '':
        if args.dbscan:
            filename = 'PROTOTYPE_DBSCAN.txt'
        else:
            filename = 'PROTOTYPE.txt'
        model.eval('prototype', os.path.join(args.save_dir, 'categorization', filename), args.dbscan)
    if args.method == 'kde':
        if args.dbscan:
            filename = 'PROTOTYPE_KDE_DBSCAN.txt'
        else:
            filename = 'PROTOTYPE_KDE.txt'
        model.eval('kde', os.path.join(args.save_dir, 'categorization', filename), args.dbscan)
