from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from dbscan import *
from imagenet_utils import *
from scipy.stats import gaussian_kde, multivariate_normal
import random
from tqdm import tqdm

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

    def get_gaussian(self, encodings):
        X, Y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        prototype = np.mean(encodings, axis=0)
        kernel = multivariate_normal(mean=prototype, cov=np.cov(encodings.T))
        return kernel, prototype

    def entails(self, pair, method):
        w1, w2, d = pair
        w1_encodings, w2_encodings = get_exclusive_encodings([w1, w2], os.path.join(self.model_path, 'encodings.hdf5'))
        if self.dbscan:
            w1_encodings = dbscan_filter(w1_encodings)
            w2_encodings = dbscan_filter(w2_encodings)
        if method == 'kde':
            w1_kde, w1_prototype = self.get_kde(w1_encodings)
            w2_kde, w2_prototype = self.get_kde(w2_encodings)
            return w2_kde(w1_prototype)[0] / w2_kde(w2_prototype)[0]
        else:
            w1_kde, w1_prototype = self.get_gaussian(w1_encodings)
            w2_kde, w2_prototype = self.get_gaussian(w2_encodings)
            return w2_kde.pdf(w1_prototype) / w2_kde.pdf(w2_prototype)

    def entailment(self, eval_set, save_path, method):
        if eval_set == 'wbless':
            dset = WBless()
        else:
            dset = Hyperlex()
        with open(save_path, 'w+') as f:
            for pair in tqdm(dset.pairs, total=len(dset.pairs)):
                entails = self.entails(pair, method)
                line = '{0}\t{1}\t{2}\t{3}'.format(get_word_from_label(pair[0]), get_word_from_label(pair[1]), pair[2], str(entails))
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--dbscan', action='store_true')
    parser.add_argument('--eval_set', help='override for only wbless or hyperlex', type=str, default='')
    parser.add_argument('--kde', action='store_true')
    args = parser.parse_args()

    args.model_path = 'model/vae' + str(args.latent_size)
    args.save_dir = 'results/' + str(args.latent_size)

    model = Prototype(args.model_path, args.dbscan)
    if args.dbscan:
        if args.kde:
            filename = 'PROTOTYPE_KDE_DBSCAN.txt'
        else:
            filename = 'PROTOTYPE_DBSCAN.txt'
    else:
        if args.kde:
            filename = 'PROTOTYPE_KDE.txt'
        else:
            filename = 'PROTOTYPE.txt'
    if args.kde:
        method = 'kde'
    else:
        method = 'gaussian'
    if args.eval_set == 'wbless':
        model.entailment('wbless', os.path.join(args.save_dir, 'wbless', filename), method)
    elif args.eval_set == 'hyperlex':
        model.entailment('hyperlex', os.path.join(args.save_dir, 'hyperlex', filename), method)
    else:
        model.entailment('wbless', os.path.join(args.save_dir, 'wbless', filename), method)
        model.entailment('hyperlex', os.path.join(args.save_dir, 'hyperlex', filename), method)
