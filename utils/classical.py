from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.spatial import ConvexHull, Delaunay
from tqdm import tqdm

class Classical():
    def __init__(self, model_path, dbscan=False):
        self.model_path = model_path
        self.dbscan = dbscan

    def entails(self, pair):
        w1, w2, d = pair
        w1_encodings, w2_encodings = get_exclusive_encodings([w1, w2], self.model_path)
        if self.dbscan:
            w1_encodings = dbscan_filter(w1_encodings)
            w2_encodings = dbscan_filter(w2_encodings)
        d = Delaunay(w2_encodings)
        n_entails = 0
        n = 0
        for enc in w1_encodings:
            if d.find_simplex(enc) >= 0:
                n_entails += 1
            n += 1
        return float(n_entails) / n

    def entailment(self, save_path):
        dset = Hyperlex()#WBless()
        with open(save_path, 'w+') as f:
            for pair in tqdm(dset.pairs, total=len(dset.pairs)):
                entails = self.entails(pair)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--save_path', type=str, default='results/CLASSICAL.txt')
    parser.add_argument('--entailment', action='store_true')
    args = parser.parse_args()

    c = Classical(args.model_path, args.dbscan)
    if args.entailment:
        c.entailment(args.save_path)
