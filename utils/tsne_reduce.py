import h5py
from sklearn.decomposition import PCA
import sys

input, output = sys.argv[1:3]
with h5py.File(input) as f:
    enc = f['encodings']
    emb = PCA(n_components=2).fit_transform(enc)
with h5py.File(output, 'w') as f:
    f.create_dataset('encodings', data=emb)
