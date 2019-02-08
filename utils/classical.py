from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.spatial import Delaunay
from tqdm import tqdm
import re

class Classical():
    def __init__(self, model_path, dbscan=False):
        self.model_path = model_path
        self.dbscan = dbscan

    def entails(self, pair):
        w1, w2, d = pair
        w1_encodings, w2_encodings = get_exclusive_encodings([w1, w2], os.path.join(self.model_path, 'encodings.hdf5'))
        if self.dbscan:
            w1_encodings = dbscan_filter(w1_encodings)
            w2_encodings = dbscan_filter(w2_encodings)
        n_entails = 0
        n = 0
        d = Delaunay(w2_encodings)
        for enc in w1_encodings:
            if d.find_simplex(enc) >= 0:
                n_entails += 1
            n += 1
        return float(n_entails) / n

    def entailment(self, eval_set, save_path):
        if eval_set == 'wbless':
            dset = WBless()
        else:
            dset = Hyperlex()
        with open(save_path, 'w+') as f:
            for pair in tqdm(dset.pairs, total=len(dset.pairs)):
                entails = self.entails(pair)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--eval_set', help='override for only wbless or hyperlex', type=str, default='')
    args = parser.parse_args()

    args.model_path = 'model/vae' + str(args.latent_size)
    args.save_dir = 'results/' + str(args.latent_size)

    model = Classical(args.model_path, args.dbscan)
    if args.dbscan:
        filename = 'CLASSICAL_DBSCAN.txt'
    else:
        filename = 'CLASSICAL.txt'
    if args.eval_set == 'wbless':
        model.entailment('wbless', os.path.join(args.save_dir, 'wbless', filename))
    elif args.eval_set == 'hyperlex':
        model.entailment('hyperlex', os.path.join(args.save_dir, 'hyperlex', filename))
    else:
        model.entailment('wbless', os.path.join(args.save_dir, 'wbless', filename))
        model.entailment('hyperlex', os.path.join(args.save_dir, 'hyperlex', filename))
