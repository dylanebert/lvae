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
        wbless = WBless()
        with open(save_path, 'w+') as f:
            for pair in wbless.pairs:
                entails = self.entails(pair)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                print(line)
                f.write('{0}\n'.format(line))

    def numerical_eval(self, save_path):
        true = []
        predicted = []
        with open(save_path) as f:
            for line in f:
                _, _, t, p = line.rstrip().split('\t')
                true.append(int(t))
                predicted.append(float(p))
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(true)):
            if true[i] == 1:
                if predicted[i] == 1:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if predicted[i] == 1:
                    false_positives += 1
        precision = true_positives / float(true_positives + false_positives)
        recall = true_positives / float(true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(precision, recall, f1)

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
