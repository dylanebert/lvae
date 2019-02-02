import os
from wordnet_utils import *
from imagenet_utils import *
from wbless_bridge import WBless
from dbscan import *
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np
from collections import defaultdict

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
        print(X[p_idx], Y[p_idx], Z[p_idx])
        plt.imshow(np.rot90(Z), cmap=cmap, extent=[-2, 2, -2, 2])
        plt.plot(encodings[:,0], encodings[:,1], 'k.', markersize=1, alpha=.2, label=get_word_from_label(label))
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.show()

    def prec_recall(self, path):
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        true_negatives = defaultdict(int)
        taus = np.linspace(.01, 1, num=100, endpoint=True)
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
            accuracy = (true_positives[tau] + true_negatives[tau]) / float(n)
            precision = true_positives[tau] / float(true_positives[tau] + false_positives[tau])
            recall = true_positives[tau] / float(true_positives[tau] + false_negatives[tau])
            f1 = 2 * precision * recall / (precision + recall)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            print('\t'.join([str(i) for i in [tau, accuracy, precision, recall, f1]]))
        plt.plot(taus, precisions, label='precision')
        plt.plot(taus, recalls, label='recall')
        #plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    vis = Visualization('model/vae2')
    #words = ['bird_of_prey', 'owl', 'eagle']
    #labels = [get_label_from_word(word) for word in words]
    #vis.scatter(labels, dbscan=False)

    vis.prec_recall('results/CLASSICAL_DBSCAN.txt')
