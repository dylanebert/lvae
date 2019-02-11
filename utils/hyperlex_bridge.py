import pickle
from imagenet_utils import *

class Hyperlex():
    def __init__(self):
        with open('/data/nlp/hyperlex/hyperlex_imagenet_labels.txt') as f:
            self.labels = f.read().splitlines()
        with open('/data/nlp/hyperlex/hyperlex_imagenet.p', 'rb') as f:
            self.pairs = pickle.load(f)

    def __str__(self):
        string = ''
        for (w1, w2, l1, l2, d) in self.pairs:
            string += '{0}\t{1}\t{2}\n'.format(w1, w2, d)
        return string
