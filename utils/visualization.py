import os
from wordnet_utils import *
from imagenet_utils import *
from wbless_bridge import WBless
from dbscan import *
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

class Visualization():
    def __init__(self, model_path):
        self.model_path = model_path
        self.wbless = WBless()

    def scatter(self, labels, dbscan=False):
        palette = sns.color_palette('muted', n_colors=len(labels))
        cmap = ListedColormap(sns.color_palette(palette).as_hex())
        plt.box(on=None)
        plt.grid(alpha=.5, linestyle='-', linewidth=1)
        for i, label in enumerate(labels):
            encodings = get_concept_encodings(label, self.model_path)
            if dbscan:
                encodings = dbscan_filter(encodings)
            plt.scatter(encodings[:,0], encodings[:,1], alpha=.2, color=cmap(float(i)/len(labels)), label=get_word_from_label(label))
            h = ConvexHull(encodings)
            for simplex in h.simplices:
                plt.plot(encodings[simplex, 0], encodings[simplex, 1], 'k-', color=cmap(float(i)/len(labels)))
        plt.legend()
        plt.xticks(alpha=.5)
        plt.yticks(alpha=.5)
        if dbscan:
            plt.title('Sample VAE encodings (with dbscan)')
        else:
            plt.title('Sample VAE encodings (without dbscan)')
        plt.show()

if __name__ == '__main__':
    vis = Visualization('model/vae2')
    words = ['bird_of_prey', 'owl', 'eagle', 'hawk']
    labels = [get_label_from_word(word) for word in words]
    vis.scatter(labels, dbscan=False)
    vis.scatter(labels, dbscan=True)
