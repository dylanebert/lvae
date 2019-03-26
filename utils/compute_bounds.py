from imagenet_utils import *
import os
import numpy as np
from csv_utils import *
import argparse
from sklearn.neighbors import NearestNeighbors

class Bounds:
    def __init__(self, model, label, include_outliers=False):
        self.model = model
        self.label = label
        self.compute_neighbors(include_outliers)

    def compute_neighbors(self, include_outliers):
        encodings = get_cluster_encodings(self.model, self.label, include_outliers)
        self.neighbors = NearestNeighbors(n_neighbors=2).fit(encodings)
        distances, indices = self.neighbors.kneighbors(encodings)
        self.radius = np.amax(distances)

    def in_bounds(self, encodings):
        radius_neighbors = self.neighbors.radius_neighbors(encodings, self.radius, return_distance=False)
        in_bounds = []
        for n in radius_neighbors:
            if n.size == 0:
                in_bounds.append(False)
            else:
                in_bounds.append(True)
        return in_bounds
