import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from PIL import Image
import os
from sklearn.decomposition import PCA
from os.path import join
from config import config
from sklearn.neighbors import NearestNeighbors
import shutil
from vae import VAE
import argparse

def plot_encodings():
    with h5py.File(os.path.join(config.model, 'train_encodings.h5')) as f:
        try:
            encodings = np.array(f['encodings'])
        except:
            encodings = np.array(f['embeddings'])
        filenames = [s.decode('utf-8') for s in f['filenames']]
        labels = [s.decode('utf-8') for s in f['labels']]
    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
    cmap = ListedColormap(sns.color_palette('husl', 8).as_hex())
    colors = [cmap(.5)] * len(encodings)
    for i, x in enumerate(labels):
        if x == 'tree':
            colors[i] = 'red'
    if encodings.shape[1] > 2:
        encodings = PCA(n_components=2).fit_transform(encodings)
    plt.scatter(encodings[:,0], encodings[:,1], s=5, picker=5, c=colors)
    #plt.xlim([-2, 2])
    #plt.ylim([-2, 2])

    def onpick(event):
        ind = event.ind[0]
        img = Image.open(os.path.join(config.data, 'train', filenames[ind]))
        img.show()
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--encodings', action='store_true')
args = parser.parse_args()

if args.encodings:
    plot_encodings()
