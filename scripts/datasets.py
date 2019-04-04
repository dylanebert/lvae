import os
import json
import pandas as pd
import numpy as np

concreteness = pd.read_csv('/data/nlp/concreteness.txt', sep='\t', index_col=False)
concrete_labels = concreteness[concreteness['Conc.M'] - concreteness['Conc.SD'] > 4]['Word'].values

class wbless():
    def __init__(self, filter=False):
        with open('/data/nlp/bless/WBLESS.json') as f:
            self.pairs = pd.DataFrame(data=json.loads(f.read()), columns=['WORD1', 'WORD2', 'ENTAILS'])
        self.labels = list(set(np.concatenate((self.pairs['WORD1'], self.pairs['WORD2']))))
        if filter:
            self.labels = [label for label in self.labels if label in concrete_labels]


class hyperlex():
    def __init__(self, filter=False):
        self.pairs = pd.read_csv('/data/nlp/hyperlex/hyperlex-all.txt', sep=' ', usecols=[0,1,4], index_col=False)
        self.labels = list(set(np.concatenate((self.pairs['WORD1'], self.pairs['WORD2']))))
        if filter:
            self.labels = [label for label in self.labels if label in concrete_labels]

wbless = wbless(filter=True)
hyperlex = hyperlex(filter=True)
eval_labels = list(set(wbless.labels + hyperlex.labels))
