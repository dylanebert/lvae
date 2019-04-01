import os
import pandas as pd
from datasets import eval_labels
from nltk.corpus import wordnet
import difflib
from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np
import shutil
import random

def build():
    mmid_labels = {}
    dir = '/data/nlp/mmid/dicts'
    for s in os.listdir(dir):
        df = pd.read_csv(os.path.join(dir, s), sep='\t', index_col=False, header=None, error_bad_lines=False, warn_bad_lines=False)
        for i, (label, index) in enumerate(zip(df[0], df[1])):
            try:
                label, index = str(label), int(index)
            except:
                pass
            mmid_labels[label] = os.path.join(s.replace('.tsv', ''), str(index))

    imagenet_labels = {}
    with open('/data/nlp/imagenet/synsets.txt') as f:
        imagenet_synsets = f.read().splitlines()
    for s in imagenet_synsets:
        word = wordnet.synset_from_pos_and_offset(s[0], int(s[1:])).lemmas()[0].name().replace('_', ' ')
        imagenet_labels[word] = s

    def hypernym_closure(word):
        closure = [word]
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            return closure
        def f(x):
            try:
                return x.hypernyms()[0]
            except:
                return []
        for hypernym in synsets[0].closure(f):
            closure.append(hypernym.lemmas()[0].name().replace('_', ' '))
        return list(set(closure))

    mmid_hypernyms = {}
    imagenet_hypernyms = {}
    for label in mmid_labels:
        mmid_hypernyms[label] = hypernym_closure(label)
    for label in imagenet_labels:
        imagenet_hypernyms[label] = hypernym_closure(label)

    mmid_dict = defaultdict(list)
    imagenet_dict = defaultdict(list)
    missing = []
    for label in tqdm(eval_labels, total=len(eval_labels)):
        present = False
        for s in mmid_labels:
            if label in mmid_hypernyms[s]:
                mmid_dict[label].append(mmid_labels[s])
                present = True
        for s in imagenet_labels:
            if label in imagenet_hypernyms[s]:
                imagenet_dict[label].append(imagenet_labels[s])
                present = True
        if not present:
            missing.append(label)

    with open('data/combined/mmid.json', 'w+') as f:
        f.write(json.dumps(mmid_dict))
    with open('data/combined/imagenet.json', 'w+') as f:
        f.write(json.dumps(imagenet_dict))
    with open('data/combined/missing.txt', 'w+') as f:
        f.write('\n'.join(missing))

def copy(mmid, imagenet):
    mmid_values = list(set(np.concatenate(list(mmid.values()))))
    imagenet_values = list(set(np.concatenate(list(imagenet.values()))))

    for value in tqdm(mmid_values, total=len(mmid_values)):
        source = os.path.join('/data/nlp/mmid', value.replace('index', 'scale'))
        files = [file for file in os.listdir(source) if file.endswith('.jpg')]
        random.shuffle(files)
        k = len(files) // 10
        filenames = {'train': files[:(8*k)], 'dev': files[8*k:9*k], 'test': files[9*k:]}
        for type, fnames in filenames.items():
            dir = os.path.join('/data/nlp/combined', type, value.replace('/', '_'))
            if not os.path.exists(dir):
                os.makedirs(dir)
            for fname in fnames:
                shutil.copy(os.path.join(source, fname), os.path.join(dir, fname))

    for value in tqdm(imagenet_values, total=len(imagenet_values)):
        for type in ['train', 'dev', 'test']:
            source = os.path.join('/data/nlp/imagenet', type, value)
            dest = os.path.join('/data/nlp/combined', type, value)
            shutil.copytree(source, dest)

try:
    with open('data/combined/mmid.json') as f:
        mmid = json.loads(f.read())
    with open('data/combined/imagenet.json') as f:
        imagenet = json.loads(f.read())
except:
    build()

if __name__ == '__main__':
    copy(mmid, imagenet)
