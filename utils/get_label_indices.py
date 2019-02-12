import h5py
import pickle
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
args = parser.parse_args()

indices = {}
with h5py.File(args.input) as f:
    filenames = f['filenames']
    current_word = None
    start_idx = 0
    n = 0
    for i in tqdm(range(len(filenames))):
        word = os.path.split(filenames[i])[0].decode('utf-8')
        if not word == current_word:
            if n > 0:
                indices[current_word] = (start_idx, n)
                n = 0
            start_idx = i
            current_word = word
        n += 1
    indices[current_word] = (start_idx, n)

with open(args.output, 'wb+') as f:
    pickle.dump(indices, f)
