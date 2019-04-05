import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from PIL import Image
import os

def plot_encodings(encodings):
    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
    plt.scatter(encodings[:,0], encodings[:,1], s=5)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()

def plot_clusters(encodings, clusters, filenames):
    clusters += 1
    n_clusters = np.amax(clusters)
    cmap = ListedColormap(sns.color_palette('husl', 8).as_hex())
    colors = [cmap(i / float(n_clusters)) for i in clusters]

    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
    plt.scatter(encodings[:,0], encodings[:,1], s=5, c=colors, picker=5)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    def onpick(event):
        ind = event.ind[0]
        if filenames[ind][0] == 'n':
            img = Image.open(os.path.join('/data/nlp/imagenet/train', filenames[ind]))
        else:
            img = Image.open(os.path.join('/data/nlp/mmid', filenames[ind].replace('_', '/')))
        img.show()
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

word = sys.argv[1]
encodings = np.array(pd.read_csv('model/combined/csv/train_encodings/' + word, header=None).values, dtype=float)
clusters = np.squeeze(np.array(pd.read_csv('model/combined/csv/train_clusters/' + word, header=None).values, dtype=int))
filenames = np.squeeze(np.array(pd.read_csv('model/combined/csv/train_filenames/' + word, header=None).values))
plot_clusters(encodings, clusters, filenames)
