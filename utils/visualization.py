import os
from wordnet_utils import *
from imagenet_utils import *
from wbless_bridge import WBless
from dbscan import *
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, multivariate_normal
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
from collections import defaultdict
import argparse

class Visualization():
    def __init__(self, model_path):
        self.model_path = model_path
        self.wbless = WBless()

    def scatter(self, labels, dbscan=False):
        palette = sns.color_palette('muted', n_colors=len(labels))
        cmap = ListedColormap(sns.color_palette(palette).as_hex())
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        encodings = get_exclusive_encodings(labels, self.model_path)
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
        if dbscan:
            plt.title('Sample VAE encodings (with dbscan)')
        else:
            plt.title('Sample VAE encodings (without dbscan)')
        plt.show()

    def kde(self, label, dbscan=False):
        cmap = sns.cubehelix_palette(rot=.1, light=1, as_cmap=True)
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        encodings = get_concept_encodings(label, self.model_path)
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
        encodings = get_concept_encodings(label, self.model_path)
        if dbscan:
            encodings = dbscan_filter(encodings)
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        prototype = np.mean(encodings, axis=0)
        kernel = multivariate_normal(mean=prototype, cov=np.cov(encodings.T))
        Z = np.reshape(kernel.pdf(pos), X.shape)
        plt.imshow(np.rot90(Z), cmap=cmap, extent=[-2, 2, -2, 2])
        #plt.contourf(X, Y, kernel.pdf(pos), cmap=cmap)
        plt.plot(encodings[:,0], encodings[:,1], 'k.', markersize=1, alpha=.1, label=get_word_from_label(label))
        plt.plot(prototype[0], prototype[1], 'wx')
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.title('Parametric Gaussian for {0}'.format(get_word_from_label(label)))
        plt.show()

    def prec_recall(self, path, endpoint=False, title=''):
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        true_negatives = defaultdict(int)
        taus = np.linspace(.01, 1, num=100, endpoint=endpoint)
        n = 0
        with open(path) as f:
            for line in f:
                n += 1
                w1, w2, t, p = line.rstrip().split('\t')
                t = int(t)
                p = float(p)
                for tau in taus:
                    if p >= tau:
                        if t == 1:
                            true_positives[tau] += 1
                        else:
                            false_positives[tau] += 1
                    else:
                        if t == 1:
                            false_negatives[tau] += 1
                        else:
                            true_negatives[tau] += 1
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        for tau in taus:
            try:
                accuracy = (true_positives[tau] + true_negatives[tau]) / float(n)
            except:
                accuracy = 0
            try:
                precision = true_positives[tau] / float(true_positives[tau] + false_positives[tau])
            except:
                precision = 0
            try:
                recall = true_positives[tau] / float(true_positives[tau] + false_negatives[tau])
            except:
                recall = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except:
                f1 = 0
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print('\t'.join([str(i) for i in [tau, accuracy, precision, recall, f1]]))
        plt.plot(taus, precisions, label='precision')
        plt.plot(taus, recalls, label='recall')
        #plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel(r'$\tau$')
        plt.ylabel('precision/recall')
        plt.title(title)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--save_path', type=str, default='results/PROTOTYPE.txt')
    parser.add_argument('--word', type=str, default='bird_of_prey')
    parser.add_argument('--prec_recall', type=str, default='')
    parser.add_argument('--endpoint', action='store_true')
    parser.add_argument('--kde', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--scatter', action='store_true')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    if not args.prec_recall == '':
        vis.prec_recall(args.save_path, endpoint=args.endpoint, title=args.prec_recall)
    if args.kde:
        vis.kde(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.gaussian:
        vis.gaussian(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.scatter:
        vis.scatter([get_label_from_word(args.word)], dbscan=args.dbscan)
