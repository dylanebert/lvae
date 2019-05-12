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
sys.path.append('/home/dylan/Documents/lvae')
from vae import VAE

def plot_encodings(encodings, filenames, labels, clusters=None, reconstructions=False):
    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
    cmap = ListedColormap(sns.color_palette('husl', 8).as_hex())
    if not clusters == None:
        clusters = np.array(clusters) + 1
        n = np.amax(clusters)
        colors = [cmap(i / float(n)) for i in clusters]
    else:
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

def plot_manifold():
    vae = VAE(config.model, config.data, config.latent_size)
    vae.load_weights()
    n = 30
    k = 64
    figure = np.zeros((k*n, k*n, 3))
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi] + [0] * (config.latent_size - 2)])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(k, k, 3)
            figure[i*k:(i+1)*k, j*k:(j+1)*k] = digit
    plt.figure(figsize=(10,10))
    start_range = k // 2
    end_range = (n - 1) * k + start_range + 1
    pixel_range = np.arange(start_range, end_range, k)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.imshow(figure)
    plt.show()

config.data = config.data.replace('embeddings', '/data/nlp')
with h5py.File(config.model + '/train_encodings.h5') as f:
    encodings = np.array(f['encodings'])
    filenames = [s.decode('utf-8') for s in f['filenames']]
    labels = [s.decode('utf-8') for s in f['labels']]
'''cluster_dim = str(sys.argv[1])
with open(join(config.model, 'csv', 'train', 'clusters_' + cluster_dim + '.csv')) as f:
    clusters = np.array(f.read().splitlines(), dtype=int).tolist()'''
plot_encodings(encodings, filenames, labels)
#plot_manifold()
