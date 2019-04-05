import pandas as pd
import h5py
import os
from datasets import eval_labels
from builder import mmid, imagenet
import numpy as np
from tqdm import tqdm
import random

def get_df(type):
    with h5py.File('model/combined/' + type + '_encodings.h5') as f:
        encodings = np.array(f['encodings'])
    with h5py.File('data/combined/' + type + '.h5') as f:
        labels = np.array(f['labels']).astype(str).tolist()
        filenames = np.array(f['filenames']).astype(str).tolist()
    encoding_columns = [np.squeeze(x) for x in np.hsplit(encodings, encodings.shape[1])]
    data = {}
    for i, v in enumerate(encoding_columns):
        data[str(i)] = v
    data['label'] = labels
    data['filename'] = filenames
    return pd.DataFrame(data=data)

def extract(type):
    df = get_df(type)
    for label in tqdm(eval_labels, total=len(eval_labels)):
        s = []
        if label in mmid:
            s += mmid[label]
        if label in imagenet:
            s += imagenet[label]
        s = [i.replace('/', '_') for i in s]
        entries = df[df['label'].isin(s)]
        temp = list(set(entries['label'].values))
        label_to_val = {label: temp.index(label) for label in temp}
        entries['label'].replace(label_to_val, inplace=True)
        if len(entries) > 10000:
            entries = entries.sample(10000)
        entries.to_csv('model/combined/csv/' + type + '_encodings/' + label, columns=['0', '1', 'label'], index=False, header=None)
        entries.to_csv('model/combined/csv/' + type + '_filenames/' + label, columns=['filename'], index=False, header=None)

if __name__ == '__main__':
    extract('train')
