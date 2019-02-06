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
            train = get_concept_encodings(label, self.model_path)
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
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--classical', action='store_true')
    parser.add_argument('--prototype', action='store_true')
    parser.add_argument('--kde', action='store_true')
    parser.add_argument('--save_path', type=str, default='results/categorization/CLASSICAL.txt')
    args = parser.parse_args()

    cat = Categorization(args.model_path)
    if args.classical:
        cat.eval('classical', args.save_path, args.dbscan)
    if args.prototype:
        cat.eval('prototype', args.save_path, args.dbscan)
    if args.kde:
        cat.eval('kde', args.save_path, args.dbscan)
