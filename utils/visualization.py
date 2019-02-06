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

    def prec_recall(self, dbscan=False):
        precision_dict = defaultdict(list)
        recall_dict = defaultdict(list)
        if dbscan:
            paths = ['results/wbless/CLASSICAL_DBSCAN.txt', 'results/wbless/PROTOTYPE_DBSCAN.txt', 'results/wbless/PROTOTYPE_KDE_DBSCAN.txt']
        else:
            paths = ['results/wbless/CLASSICAL.txt', 'results/wbless/PROTOTYPE.txt', 'results/wbless/PROTOTYPE_KDE.txt']
        methods = ['Classical', 'Prototype (parametric)', 'Prototype (non-parametric)']
        for method, path in zip(methods, paths):
            with open(path) as f:
                true = []
                predicted = []
                for line in f:
                    w1, w2, t, p = line.rstrip().split('\t')
                    true.append(int(t))
                    predicted.append(float(p))
                precision, recall, thresholds = precision_recall_curve(true, predicted)
                precision_dict[method] = precision
                recall_dict[method] = recall

        plt.figure(figsize=(8,6))
        for method in methods:
            precision = precision_dict[method]
            recall = recall_dict[method]
            plt.plot(recall, precision, label=method)
        plt.ylim([.64, 1])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        if dbscan:
            plt.title('Precision vs recall (with DBSCAN)')
        else:
            plt.title('Precision vs recall (without DBSCAN)')
        plt.show()

    def hyperlex(self, dbscan=False):
        if dbscan:
            paths = ['results/hyperlex/CLASSICAL_DBSCAN.txt', 'results/hyperlex/PROTOTYPE_DBSCAN.txt', 'results/hyperlex/PROTOTYPE_KDE_DBSCAN.txt']
        else:
            paths = ['results/hyperlex/CLASSICAL.txt', 'results/hyperlex/PROTOTYPE.txt', 'results/hyperlex/PROTOTYPE_KDE.txt']
        methods = ['Classical', 'Prototype (parametric)', 'Prototype (non-parametric)']
        for method, path in zip(methods, paths):
            with open(path) as f:
                true = []
                predicted = []
                for line in f:
                    w1, w2, t, p = line.rstrip().split('\t')
                    true.append(float(t))
                    predicted.append(float(p))
            print(method, spearmanr(true, predicted), mean_absolute_error(true, predicted), math.sqrt(mean_squared_error(true, predicted)))

    def categorization(self, dbscan=False):
        #Classical - accuracy only
        y_true = []
        y_pred = []
        if dbscan:
            path = 'results/categorization/CLASSICAL_DBSCAN.txt'
        else:
            path = 'results/categorization/CLASSICAL.txt'
        with open(path) as f:
            for line in f:
                t, p = line.rstrip().split('\t')
                y_true.append(int(float(t)))
                y_pred.append(int(float(p)))
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for t, p in zip(y_true, y_pred):
            if p == 1:
                if t == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if t == 1:
                    false_negatives += 1
        prec = true_positives / float(true_positives + false_positives)
        rec = true_positives / float(true_positives + false_negatives)
        print('Classical', accuracy_score(y_true, y_pred), prec, rec, 2 * prec * rec / (prec + rec))

        #Prototype
        precision_dict = defaultdict(list)
        recall_dict = defaultdict(list)
        threshold_dict = defaultdict(list)
        if dbscan:
            paths = ['results/categorization/PROTOTYPE_DBSCAN.txt', 'results/categorization/PROTOTYPE_KDE_DBSCAN.txt']
        else:
            paths = ['results/categorization/PROTOTYPE.txt', 'results/categorization/PROTOTYPE_KDE.txt']
        methods = ['Prototype (parametric)', 'Prototype (non-parametric)']
        for method, path in zip(methods, paths):
            y_true = []
            y_pred = []
            with open(path) as f:
                for line in f:
                    t, p = line.rstrip().split('\t')
                    y_true.append(int(float(t)))
                    y_pred.append(float(p))
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                precision_dict[method] = precision
                recall_dict[method] = recall
                max_f1 = 0
                max_f1_idx = None
                max_prec = 0
                max_prec_idx = None
                for i, (threshold, p, r) in enumerate(list(zip(thresholds, precision, recall))):
                    f1 = 2 * p * r / (p + r)
                    if f1 > max_f1:
                        max_f1 = f1
                        max_f1_idx = i
                    if p > max_prec and r > .01:
                        max_prec = p
                        max_prec_idx = i
                y_pred_int = []
                for p in y_pred:
                    if p >= thresholds[max_f1_idx]:
                        y_pred_int.append(1)
                    else:
                        y_pred_int.append(0)
                print(method, thresholds[max_f1_idx], accuracy_score(y_true, y_pred_int), precision[max_f1_idx], recall[max_f1_idx], max_f1)
                y_pred_int = []
                for p in y_pred:
                    if p >= thresholds[max_prec_idx]:
                        y_pred_int.append(1)
                    else:
                        y_pred_int.append(0)
                rec = recall[max_prec_idx]
                print(method, thresholds[max_prec_idx], accuracy_score(y_true, y_pred_int), max_prec, rec, 2 * max_prec * rec / (max_prec + rec))
        plt.figure(figsize=(8,6))
        for method in methods:
            precision = precision_dict[method]
            recall = recall_dict[method]
            plt.plot(recall, precision, label=method)
        plt.ylim([.5, 1])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        if dbscan:
            plt.title('Categorization precision vs recall (with DBSCAN)')
        else:
            plt.title('Categorization precision vs recall (without DBSCAN)')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--word', type=str, default='bird_of_prey')
    parser.add_argument('--prec_recall', action='store_true')
    parser.add_argument('--endpoint', action='store_true')
    parser.add_argument('--kde', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--scatter', action='store_true')
    parser.add_argument('--hyperlex', action='store_true')
    parser.add_argument('--categorization', action='store_true')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    if args.prec_recall:
        vis.prec_recall(args.dbscan)
    if args.hyperlex:
        vis.hyperlex(dbscan=args.dbscan)
    if args.kde:
        vis.kde(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.gaussian:
        vis.gaussian(get_label_from_word(args.word), dbscan=args.dbscan)
    if args.scatter:
        vis.scatter([get_label_from_word(args.word)], dbscan=args.dbscan)
    if args.categorization:
        vis.categorization(dbscan=args.dbscan)
