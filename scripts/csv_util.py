import pandas as pd
import h5py
import os
from datasets import eval_labels
from builder import mmid, imagenet
import numpy as np
from tqdm import tqdm
import random
from config import config

def get_df():
    with h5py.File(config.model + '/train_encodings.h5') as f:
        encodings = np.array(f['encodings'])
    with h5py.File(config.data + '/train.h5') as f:
        labels = np.array(f['labels']).astype(str).tolist()
        filenames = np.array(f['filenames']).astype(str).tolist()
    encoding_columns = [np.squeeze(x) for x in np.hsplit(encodings, encodings.shape[1])]
    data = {}
    for i, v in enumerate(encoding_columns):
        data[str(i)] = v
    data['label'] = labels
    ki = list(data.keys())
    data['filename'] = filenames
    return pd.DataFrame(data=data), ki

def extract(label_sets):
    for dir in [config.model + '/csv/train_encodings', config.model + '/csv/train_filenames', config.model + '/csv/train_clusters']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    df, ki = get_df()
    for label, s in tqdm(label_sets.items(), total=len(label_sets)):
        entries = df[df['label'].isin(s)]
        temp = list(set(entries['label'].values))
        label_to_val = {label: temp.index(label) for label in temp}
        entries['label'].replace(label_to_val, inplace=True)
        if len(entries) > 10000:
            entries = entries.sample(10000)
        entries.to_csv(config.model + '/csv/train_encodings/' + label, columns=ki, index=False, header=None)
        entries.to_csv(config.model + '/csv/train_filenames/' + label, columns=['filename'], index=False, header=None)

if __name__ == '__main__':
    '''label_sets = {}
    for label in eval_labels:
        s = []
        if label in mmid:
            s+= mmid[label]
        if label in imagenet:
            s += imagenet[label]
        s = [i.replace('/', '_') for i in s]
        label_sets[label] = s'''
    label_sets = {k: [k] for k in ['Boots', 'Sandals', 'Shoes', 'Slippers']}
    extract(label_sets)
