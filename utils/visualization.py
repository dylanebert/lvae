import os
from wordnet_utils import *
from imagenet_utils import *
from wbless_bridge import WBless
from dbscan import *
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, multivariate_normal, spearmanr
from sklearn.metrics import precision_recall_curve, r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
from collections import defaultdict
import argparse
import math

class Visualization():
    def __init__(self, model_path):
        self.model_path = model_path

    def scatter(self, labels, dbscan=False, save_path='', title=None):
        palette = sns.color_palette('muted', n_colors=len(labels))
        cmap = ListedColormap(sns.color_palette(palette).as_hex())
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        encodings = get_exclusive_encodings(labels, os.path.join(self.model_path, 'encodings_2d.hdf5'))
        for i, label in enumerate(labels):
            enc = encodings[i]
            if dbscan:
                enc = dbscan_filter(enc)
            plt.scatter(enc[:,0], enc[:,1], alpha=.2, color=cmap(float(i)/len(labels)), label=get_word_from_label(label))
            h = ConvexHull(enc)
            for simplex in h.simplices:
                plt.plot(enc[simplex, 0], enc[simplex, 1], 'k-', color=cmap(float(i)/len(labels)))
        plt.legend()
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        if title == None:
            if dbscan:
                plt.title('Sample VAE encodings (with dbscan)')
            else:
                plt.title('Sample VAE encodings (without dbscan)')
        else:
            plt.title(title)
        if not save_path == '':
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()

    def kde(self, label, dbscan=False):
        cmap = sns.cubehelix_palette(rot=.1, light=1, as_cmap=True)
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        encodings = get_concept_encodings(label, os.path.join(self.model_path, 'encodings_2d.hdf5'))
        if dbscan:
            encodings = dbscan_filter(encodings)
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(encodings.T)
        Z = np.reshape(kernel(pos), X.shape)
        p_idx = np.unravel_index(Z.argmax(), Z.shape)
        Z = Z / Z[p_idx]
        plt.imshow(np.rot90(Z), cmap=cmap, extent=[-2, 2, -2, 2])
        plt.plot(encodings[:,0], encodings[:,1], 'k.', markersize=1, alpha=.1, label=get_word_from_label(label))
        plt.plot(X[p_idx], Y[p_idx], 'wx')
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.title('Gaussian KDE of {0}'.format(get_word_from_label(label)))
        plt.show()

    def gaussian(self, label, dbscan=False):
        cmap = sns.cubehelix_palette(rot=.1, light=1, as_cmap=True)
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        encodings = get_concept_encodings(label, os.path.join(self.model_path, 'encodings_2d.hdf5'))
        if dbscan:
            encodings = dbscan_filter(encodings)
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        prototype = np.mean(encodings, axis=0)
        kernel = multivariate_normal(mean=prototype, cov=np.cov(encodings.T))
        Z = np.reshape(kernel.pdf(pos), X.shape)
        plt.imshow(np.rot90(Z), cmap=cmap, extent=[-2, 2, -2, 2])
        plt.plot(encodings[:,0], encodings[:,1], 'k.', markersize=1, alpha=.1, label=get_word_from_label(label))
        plt.plot(prototype[0], prototype[1], 'wx')
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.title('Parametric Gaussian for {0}'.format(get_word_from_label(label)))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--word', type=str, default='bird_of_prey')
    parser.add_argument('--word2', type=str, default='hawk')
    parser.add_argument('--kde', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--scatter', action='store_true')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    if args.kde:
        vis.kde(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.gaussian:
        vis.gaussian(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.scatter:
        vis.scatter([get_label_from_word(args.word), get_label_from_word(args.word2)], dbscan=args.dbscan, save_path=args.save_path)
