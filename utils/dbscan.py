import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_filter(encodings, eps=.05, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(encodings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return dbscan_filter(encodings, eps=eps*2, min_samples=min_samples)
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
