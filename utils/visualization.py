import os
from wordnet_utils import *
from imagenet_utils import *
from csv_utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
import seaborn as sns
import numpy as np
import argparse
from sklearn.decomposition import PCA
import pandas as pd
from PIL import Image
from compute_bounds import Bounds

class Visualization():
    def __init__(self, model):
        self.model = model

    def show_encodings(self, label):
        encodings, labels = read_csv(os.path.join(self.model, 'csv', label))
        clusters = read_clusters(os.path.join(self.model, 'csv', label + '_clusters'))
        filenames = read_filenames(os.path.join(self.model, 'csv', label + '_filenames'))
        clusters += 1
        n_clusters = np.amax(clusters) + 1
        cmap = ListedColormap(sns.color_palette('husl', 8).as_hex())
        colors = [cmap(i / float(n_clusters)) for i in clusters]

        fig = plt.figure()
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)

        if encodings.shape[1] > 2:
            encodings = PCA(n_components=2).fit_transform(encodings)
        else:
            bounds = Bounds(self.model, label)
            r = bounds.radius
            X, Y = np.mgrid[-2.5:2.5:.01, -2.5:2.5:.01]
            pos = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(bounds.in_bounds(pos.T), X.shape)
            plt.contourf(X, Y, Z, cmap='Greys')

        plt.scatter(encodings[:,0], encodings[:,1], c=colors, s=5, picker=5)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])

        def onpick(event):
            ind = event.ind[0]
            img = Image.open(os.path.join('/data/nlp/imagenet/train', filenames[ind]))
            img.show()
        fig.canvas.mpl_connect('pick_event', onpick)

        plt.show()

    def show_entailment(self, a, b):
        a_encodings, a_filenames = get_concept_encodings(a, self.model, 'dev', include_filenames=True)
        b_encodings, _ = read_csv(os.path.join(self.model, 'csv', b))
        b_filenames = read_filenames(os.path.join(self.model, 'csv', b + '_filenames'))
        colors = ['black'] * b_encodings.shape[0] + ['red'] * a_encodings.shape[0]
        sizes = [4] * b_encodings.shape[0] + [10] * a_encodings.shape[0]
        encodings = np.vstack((b_encodings, a_encodings))
        a_filenames = [os.path.join('/data/nlp/imagenet/dev', s) for s in a_filenames]
        b_filenames = [os.path.join('/data/nlp/imagenet/train', s) for s in b_filenames]
        filenames = b_filenames + a_filenames

        fig = plt.figure()
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)

        if a_encodings.shape[1] > 2:
            encodings = PCA(n_components=2).fit_transform(encodings)
        else:
            a_bounds, b_bounds = Bounds(self.model, a, include_outliers=True), Bounds(self.model, b, include_outliers=True)
            a_r, b_r = a_bounds.radius, b_bounds.radius
            X, Y = np.mgrid[-2.5:2.5:.01, -2.5:2.5:.01]
            pos = np.vstack([X.ravel(), Y.ravel()])
            a_Z = np.reshape(a_bounds.in_bounds(pos.T), X.shape)
            b_Z = np.reshape(b_bounds.in_bounds(pos.T), X.shape)
            plt.contourf(X, Y, b_Z, cmap='Greys', alpha=.25)
            plt.contourf(X, Y, a_Z, cmap='Reds', alpha=.25)

            a_train_encodings = get_concept_encodings(a, self.model, 'train')
            a_h, b_h = ConvexHull(a_train_encodings), ConvexHull(b_encodings)
            for simplex in b_h.simplices:
                plt.plot(b_encodings[simplex, 0], b_encodings[simplex, 1], 'k-')
            for simplex in a_h.simplices:
                plt.plot(a_train_encodings[simplex, 0], a_train_encodings[simplex, 1], 'r-')

        plt.scatter(encodings[:,0], encodings[:,1], c=colors, s=sizes, picker=5)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])

        def onpick(event):
            ind = event.ind[0]
            img = Image.open(filenames[ind])
            img.show()
        fig.canvas.mpl_connect('pick_event', onpick)

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    #vis.show_encodings(get_label_from_word('bird_of_prey'))
    #vis.show_encodings(get_label_from_word('great_gray_owl'))
    vis.show_entailment(get_label_from_word('great_gray_owl'), get_label_from_word('bird_of_prey'))
