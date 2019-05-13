from sklearn.neighbors import NearestNeighbors
import h5py
import numpy as np
import os
from config import config
from sklearn.decomposition import PCA

path = os.path.join(config.model, 'train_encodings.h5')
with h5py.File(path) as f:
    try:
        encodings = np.array(f['encodings'][:10000])
    except:
        encodings = np.array(f['embeddings'][:10000])
    filenames = [s.decode('utf-8') for s in f['filenames'][:10000]]
    labels = [s.decode('utf-8') for s in f['labels'][:10000]]

nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(encodings)
distances, indices = nbrs.kneighbors(encodings)

n, k = 0, 0
for idx in range(len(labels)):
    for i, x in enumerate(indices[idx]):
        if not i == 0:
            if labels[x] == labels[idx]:
                k += 1
            n += 1

print(k / float(n))
