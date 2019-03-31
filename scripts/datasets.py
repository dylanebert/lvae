import os
import json
import pandas as pd
import numpy as np

class wbless():
    def __init__(self):
        with open('/data/nlp/bless/WBLESS.json') as f:
            self.pairs = pd.DataFrame(data=json.loads(f.read()), columns=['WORD1', 'WORD2', 'ENTAILS'])
        self.labels = list(set(np.concatenate((self.pairs['WORD1'], self.pairs['WORD2']))))

class hyperlex():
    def __init__(self):
        self.pairs = pd.read_csv('/data/nlp/hyperlex/hyperlex-all.txt', sep=' ', usecols=[0,1,4], index_col=False)
        self.labels = list(set(np.concatenate((self.pairs['WORD1'], self.pairs['WORD2']))))

wbless = wbless()
hyperlex = hyperlex()
eval_labels = list(set(wbless.labels + hyperlex.labels))
