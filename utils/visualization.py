import os
from wordnet_utils import *
from imagenet_utils import *
from csv_utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import argparse
from sklearn.decomposition import PCA

class Visualization():
    def __init__(self, model_path):
        self.model_path = model_path

    def show_encodings(self, label):
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        data, labels = read_csv_with_labels(os.path.join(self.model_path, 'csv', label))
        n_labels = np.amax(labels)
        palette = sns.color_palette('muted')
        cmap = ListedColormap(sns.color_palette(palette).as_hex())
        colors = [cmap(i / float(n_labels)) for i in labels]
        #enc = data
        pca = PCA(n_components=2)
        enc = pca.fit_transform(data)
        plt.scatter(enc[:,0], enc[:,1], alpha=.5, c=colors)
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.show()
        plt.clf()

    def show_clusters(self, label):
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        data, labels = read_csv_with_labels(os.path.join(self.model_path, 'csv', label))
        clusters = read_clusters(os.path.join(self.model_path, 'csv', label + '_clusters'))
        n_colors = np.amax(clusters) + 1
        palette = sns.color_palette('muted')
        cmap = ListedColormap(sns.color_palette(palette).as_hex())
        colors = [cmap((i + 1) / float(n_colors)) for i in clusters]
        #enc = data
        pca = PCA(n_components=2)
        enc = pca.fit_transform(data)
        plt.scatter(enc[:,0], enc[:,1], alpha=.5, c=colors)
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        plt.show()
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae64')
    parser.add_argument('-a', type=str, default='bird_of_prey')
    args = parser.parse_args()

    vis = Visualization(args.model_path)
    vis.show_encodings(args.a)
    vis.show_clusters(args.a)
