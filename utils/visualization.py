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

    def compare(self, label, dbscan=False):
        cmap = sns.cubehelix_palette(rot=.1, light=1, as_cmap=True)
        encodings = get_concept_encodings(label, self.model_path, 'train', reduced=True)
        if dbscan:
            encodings = dbscan_filter(encodings)
        size = 2
        X, Y = np.mgrid[-size:size:.01, -size:size:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        prototype = np.mean(encodings, axis=0)
        kernel = multivariate_normal(mean=prototype, cov=np.cov(encodings.T))
        Z = np.reshape(kernel.pdf(pos), X.shape)
        plt.figure(figsize=(10,10))
        plt.imshow(np.rot90(Z), cmap=cmap, extent=[-size, size, -size, size])
        plt.plot(encodings[:,0], encodings[:,1], 'k.', markersize=2, alpha=.2, label=get_word_from_label(label))
        h = ConvexHull(encodings)
        for simplex in h.simplices:
            plt.plot(encodings[simplex, 0], encodings[simplex, 1], 'k-', color='gray')
        plt.xlim([-size, size])
        plt.ylim([-size, size])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.title('Birds of Prey, Classical vs Prototype method')
        plt.show()

    def compare2(self, l1, l2, dbscan=False):
        cmap = sns.cubehelix_palette(rot=.1, light=1, dark=.2, as_cmap=True)
        cmap2 = sns.cubehelix_palette(rot=-.2, light=1, dark=.1, as_cmap=True)
        w1_encodings, w2_encodings = get_exclusive_encodings([l1, l2], self.model_path, reduced=True)
        if dbscan:
            w1_encodings = dbscan_filter(w1_encodings)
            w2_encodings = dbscan_filter(w2_encodings)
        size = 1.5
        X, Y = np.mgrid[-size:size:.01, -size:size:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        prototype = np.mean(w1_encodings, axis=0)
        prototype2 = np.mean(w2_encodings, axis=0)
        kernel = multivariate_normal(mean=prototype, cov=np.cov(w1_encodings.T))
        kernel2 = multivariate_normal(mean=prototype2, cov=np.cov(w2_encodings.T))
        Z = np.reshape(kernel.pdf(pos), X.shape) / kernel.pdf(prototype)
        Z2 = np.reshape(kernel2.pdf(pos), X.shape) / kernel2.pdf(prototype2)
        plt.figure(figsize=(10,10))
        im1 = cmap(np.rot90(Z))
        im2 = cmap2(np.rot90(Z2))
        plt.imshow(im1 * im2, extent=[-size, size, -size, size])
        plt.plot(w1_encodings[:,0], w1_encodings[:,1], 'k.', markersize=2, alpha=.3, label=get_word_from_label(l1))
        plt.plot(w2_encodings[:,0], w2_encodings[:,1], 'w.', markersize=2, alpha=1, label=get_word_from_label(l2))
        h = ConvexHull(w1_encodings)
        h2 = ConvexHull(w2_encodings)
        for simplex in h.simplices:
            plt.plot(w1_encodings[simplex, 0], w1_encodings[simplex, 1], 'k-', alpha=.8)
        for simplex in h2.simplices:
            plt.plot(w2_encodings[simplex, 0], w2_encodings[simplex, 1], 'k-', alpha=.8)
        plt.xlim([-size, size])
        plt.ylim([-size, size])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.title('Birds of Prey, Classical vs Prototype method')
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
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--compare2', action='store_true')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    if args.kde:
        vis.kde(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.gaussian:
        vis.gaussian(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.scatter:
        vis.scatter([get_label_from_word(args.word), get_label_from_word(args.word2)], dbscan=args.dbscan, save_path=args.save_path)
    if args.compare:
        vis.compare(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.compare2:
        vis.compare2(get_label_from_word(args.word), get_label_from_word(args.word2), dbscan=args.dbscan)
