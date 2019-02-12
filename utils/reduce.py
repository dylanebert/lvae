import h5py
from sklearn.decomposition import PCA
import sys
import os

model_path = sys.argv[1]
encodings = os.path.join(model_path, 'encodings.hdf5')
encodings_dev = os.path.join(model_path, 'dev_encodings.hdf5')
encodings_test = os.path.join(model_path, 'test_encodings.hdf5')
with h5py.File(encodings) as f:
    enc = f['encodings']
    pca = PCA(n_components=2)
    pca.fit(enc)
    emb = pca.transform(enc)
with h5py.File(os.path.join(model_path, 'encodings_2d.hdf5'), 'w') as f:
    f.create_dataset('encodings', data=emb)
with h5py.File(encodings_dev) as f:
    enc_dev = f['encodings']
    emb_dev = pca.transform(enc_dev)
with h5py.File(os.path.join(model_path, 'dev_encodings_2d.hdf5'), 'w') as f:
    f.create_dataset('encodings', data=emb_dev)
with h5py.File(encodings_test) as f:
    enc_test = f['encodings']
    emb_test = pca.transform(enc_test)
with h5py.File(os.path.join(model_path, 'test_encodings_2d.hdf5'), 'w') as f:
    f.create_dataset('encodings', data=emb_test)
