from wordnet_utils import *
from imagenet_utils import *
import h5py
from tqdm import tqdm

dog_label = get_label_from_word('bird_of_prey')
dogs = get_leaves(dog_label)
encodings = {}
filenames = {}
with h5py.File('model/vae64/test_encodings.hdf5') as f:
    with h5py.File('test.hdf5') as g:
        for label in dogs:
            i, n = test_indices[label]
            encodings[label] = f['encodings'][i:i+n]
            filenames[label] = [s.decode('utf-8') for s in g['filenames'][i:i+n]]

with open('model/vae64/normals.p', 'rb') as f:
    normals = pickle.load(f)

def all():
    with open('results/birds_64.txt', 'w+') as f:
        header = 'filename\tlabel\tbop_prototypicality\tself_prototypicality\n'
        f.write(header)
        for label in tqdm(dogs, total=len(dogs)):
            encs = encodings[label]
            fnames = filenames[label]
            word = get_word_from_label(label)
            for enc, fname in zip(encs, fnames):
                dog_prototype = normals[dog_label].pdf(enc) / normals[dog_label].pdf(normals[dog_label].mean)
                self_prototype = normals[label].pdf(enc) / normals[label].pdf(normals[label].mean)
                f.write('\t'.join([str(i) for i in [fname, word, dog_prototype, self_prototype]]))
                f.write('\n')

def means():
    with open('results/birds_means_64.txt', 'w+') as f:
        header = 'label\tprototypicality\n'
        f.write(header)
        for label in tqdm(dogs, total=len(dogs)):
            encs = encodings[label]
            enc = np.mean(encs, axis=0)
            dog_prototype = normals[dog_label].pdf(enc) / normals[dog_label].pdf(normals[dog_label].mean)
            f.write('{0}\t{1}\n'.format(get_word_from_label(label), str(dog_prototype)))

all()
means()
