from wbless_bridge import WBless
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.spatial import ConvexHull, Delaunay

class Classical():
    def __init__(self, model_path, dbscan=False):
        self.model_path = model_path
        self.dbscan = dbscan

    def entails(self, pair):
        w1, w2, d = pair
        w1_encodings = get_concept_encodings(w1, self.model_path)
        w2_encodings = get_concept_encodings(w2, self.model_path)
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

    def entailment(self):
        wbless = WBless()
        with open('results/classical.txt', 'w+') as f:
            for pair in wbless.pairs:
                entails = self.entails(pair)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                print(line)
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('--dbscan', action='store_true')
    args = parser.parse_args()

    c = Classical(args.model_path, args.dbscan)
    c.entailment()
