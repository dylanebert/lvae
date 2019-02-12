from wordnet_utils import *
from imagenet_utils import *
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from tqdm import tqdm

def point_to_line(point, start, end):
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_vec_unit = line_vec / line_len
    point_vec = point - start
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_vec_unit, point_vec_scaled)
    if t < 0.:
        t = 0.
    if t > 1.:
        t = 1.
    nearest = line_vec * t
    dist = np.linalg.norm(point_vec - nearest)
    nearest = start + nearest
    return (dist, nearest)

def point_in_poly(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

birds_of_prey, filenames = get_concept_encodings(get_label_from_word('bird_of_prey'), 'model/vae2', 'train', stacked=False, reduced=True, include_filenames=True)
owl_label = get_label_from_word('great_grey_owl')
owl_encodings = birds_of_prey[owl_label]
owl_filenames = filenames[owl_label]
birds_of_prey_encodings = np.concatenate(list(birds_of_prey.values()), axis=0)
birds_of_prey_filenames = np.concatenate(list(filenames.values()), axis=0)

with open('model/vae64/hulls_2d.p', 'rb') as f:
    hulls = pickle.load(f)
owl_hull = hulls[owl_label]
with open('model/vae64/normals_2d.p', 'rb') as f:
    normals = pickle.load(f)
owl_normal = normals[owl_label]
owl_boundary = ConvexHull(owl_encodings)

point_dist = []
for enc in birds_of_prey_encodings:
    dist_list = []
    for v_idx in range(len(owl_boundary.vertices)):
        v1 = owl_boundary.vertices[v_idx - 1]
        v2 = owl_boundary.vertices[v_idx]
        start = owl_encodings[v1]
        end = owl_encodings[v2]
        temp = point_to_line(enc, start, end)
        dist_list.append(temp[0])
    inside = point_in_poly(enc[0], enc[1], owl_encodings[owl_boundary.vertices])
    if inside == True:
        dist_temp = -1. * min(dist_list)
    else:
        dist_temp = min(dist_list)
    point_dist.append(dist_temp)

labels = [get_word_from_label(label) for label in birds_of_prey.keys()]
with open('results/investigation.txt', 'w+') as f:
    f.write('filename\tlabel\tdist\tp\n')
    for p_idx in tqdm(range(len(birds_of_prey_encodings))):
        pt = birds_of_prey_encodings[p_idx, :]
        filename = birds_of_prey_filenames[p_idx]
        pt[1] = pt[1] + .01
        label = labels[p_idx // 1200]
        dist = point_dist[p_idx]
        prototypicality = owl_normal.pdf(pt)
        f.write('{0}\n'.format('\t'.join([str(i) for i in [filename.decode('utf-8'), label, dist, prototypicality]])))

'''plt.figure(figsize=(12,10))
plt.plot(owl_encodings[:, 0], owl_encodings[:, 1], 'k.', markersize=2, alpha=.5)
for simplex in owl_boundary.simplices:
    plt.plot(owl_encodings[simplex, 0], owl_encodings[simplex, 1], 'k-')
plt.plot(birds_of_prey_encodings[:, 0], birds_of_prey_encodings[:, 1], 'r.', markersize=2)
for p_idx in range(len(birds_of_prey_encodings)):
    pt = birds_of_prey_encodings[p_idx, :]
    pt[1] = pt[1] + .01
    dist = point_dist[p_idx]
    dist_label = '{0:.2f}'.format(dist)
    plt.annotate(dist_label, xy=pt)
plt.show()'''
