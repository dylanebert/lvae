from wbless_bridge import WBless
from hyperlex_bridge import Hyperlex
import argparse
from imagenet_utils import *
from tqdm import tqdm
from compute_bounds import Bounds

def classical_wbless(model, dset):
    pairs = WBless().pairs
    with open(os.path.join('results', model, 'classical_wbless_outliers.txt'), 'w+') as f:
        for (w1, w2, l1, l2, d) in pairs:
            w1_encodings = get_concept_encodings(l1, model, dset)
            w2_bounds = Bounds(model, l2, include_outliers=True)
            in_bounds = w2_bounds.in_bounds(w1_encodings)
            p = sum(in_bounds) / float(len(in_bounds))
            line = '\t'.join([str(i) for i in [w1, w2, d, p]])
            print(line)
            f.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/vae64')
    parser.add_argument('--eval', help='wbless or hyperlex', type=str, default='wbless')
    args = parser.parse_args()

    if args.eval == 'wbless':
        classical_wbless(args.model, 'dev')
