from wbless_bridge import WBless
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.stats import gaussian_kde
import random

class Prototype():
    def __init__(self, model_path, dbscan=False):
        self.model_path = model_path
        self.dbscan = dbscan

    def get_kde(self, encodings):
        if encodings.shape[0] > 10000:
            encodings = np.array(random.sample(list(encodings), 10000))
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.vstack([X.ravel(), Y.ravel()])
        kernel = gaussian_kde(encodings.T)
        Z = np.reshape(kernel(pos), X.shape)
        p_idx = np.unravel_index(Z.argmax(), Z.shape)
        return kernel, (X[p_idx], Y[p_idx])

    def entails(self, pair):
        w1, w2, d = pair
        w1_encodings, w2_encodings = get_exclusive_encodings([w1, w2], self.model_path)
        if self.dbscan:
            w1_encodings = dbscan_filter(w1_encodings)
            w2_encodings = dbscan_filter(w2_encodings)
        w1_kde, w1_prototype = self.get_kde(w1_encodings)
        w2_kde, w2_prototype = self.get_kde(w2_encodings)
        print(w1_kde(w2_prototype) / w1_kde(w1_prototype))
        return w2_kde(w1_prototype)[0] / w2_kde(w2_prototype)[0]

    def entailment(self, save_path):
        wbless = WBless()
        with open(save_path, 'w+') as f:
            for pair in wbless.pairs:
                entails = self.entails(pair)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                print(line)
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='model/vae2')
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--save_path', type=str, default='results/PROTOTYPE.txt')
    parser.add_argument('--entailment', action='store_true')
    args = parser.parse_args()

    model = Prototype(args.model_path, args.dbscan)
    if args.entailment:
        model.entailment(args.save_path)
