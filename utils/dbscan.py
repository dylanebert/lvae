import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import random

def dbscan_filter(encodings, eps=.1, min_samples=25):
    if encodings.shape[0] > 100000:
        encodings = np.array(random.sample(list(encodings), 100000))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(encodings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return dbscan_filter(encodings, eps=eps+.01, min_samples=min_samples)
    filtered = None
    max_members = 0
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        n_members = len(encodings[class_member_mask])
        if n_members > max_members:
            max_members = n_members
            filtered = encodings[class_member_mask]
    return filtered
