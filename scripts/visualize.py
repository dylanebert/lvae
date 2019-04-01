import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def scatter(encodings):
    fig = plt.figure()
    plt.box(on=None)
    plt.grid(alpha=.5, linestyle='-', linewidth=1)
