from visualization import Visualization
from hyperlex_bridge import Hyperlex
from wbless_bridge import WBless
from wordnet_utils import *
import os

visualization = Visualization('model/vae2')
hyperlex = Hyperlex()
wbless = WBless()

def vis(l1, l2, v):
    w1, w2 = get_word_from_label(l1), get_word_from_label(l2)
    title = '{0} -> {1}; {2:.2f}'.format(w1, w2, p)
    save_path = os.path.join('results/2/figures', '{0}_{1}.png'.format(w1, w2))
    visualization.scatter([l1, l2], save_path=save_path, title=title)

for (l1, l2, p) in hyperlex.pairs:
    vis(l1, l2, p)
for (l1, l2, t) in wbless.pairs:
    vis(l1, l2, t)
