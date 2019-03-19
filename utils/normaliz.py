from wordnet_utils import *
from imagenet_utils import *
import matplotlib.pyplot as plt
import PyNormaliz
import numpy as np

def convex_hull(A):
    vertices = A.T.tolist()
    vertices = [i + [1] for i in vertices]
    poly = PyNormaliz.Cone(vertices = vertices)
    hull_vertices = np.array([entry[:-1] for entry in poly.VerticesOfPolyhedron()])
    hull_indices = find_column_indices(A, hull_vertices)
    return hull_indices

label = list(train_indices.keys())[0]
encodings = get_concept_encodings(label, 'model/vae2', 'train')
hull = convex_hull(encodings)

plt.scatter(encodings[:,0], encodings[:,1])
for simplex in hull:
    plt.plot(enc[simplex, 0], enc[simplex, 1], 'k-')
plt.show()
