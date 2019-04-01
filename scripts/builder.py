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

    def hyponym_closure(label):
        closure = [label]
        synsets = wordnet.synsets(label)
        if len(synsets) == 0:
            return closure
        def f(x):
            try:
                return x.hyponyms()
            except:
                return []
        primary_sense = synsets[0]
        for hyponym in primary_sense.closure(f):
            for lemma in hyponym.lemmas():
                closure.append(lemma.name().replace('_', ' '))
        return list(set(closure))

    mmid_dict = defaultdict(list)
    imagenet_dict = defaultdict(list)
    for label in tqdm(eval_labels, total=len(eval_labels)):
        hyponyms = hyponym_closure(label)
        for hyponym in hyponyms:
            if hyponym in mmid_labels:
                mmid_dict[label].append(mmid_labels[hyponym])
            if hyponym in imagenet_labels:
                imagenet_dict[label].append(imagenet_labels[hyponym])

    with open('data/combined/mmid.json', 'w+') as f:
        f.write(json.dumps(mmid_dict))
    with open('data/combined/imagenet.json', 'w+') as f:
        f.write(json.dumps(imagenet_dict))

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
