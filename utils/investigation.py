from wordnet_utils import *
from imagenet_utils import *
from dbscan import dbscan_filter
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from tqdm import tqdm
import json
from scipy.stats import spearmanr

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

def get_dists(points, original, hull):
    point_dist = []
    for enc in points:
        dist_list = []
        for v_idx in range(len(hull.vertices)):
            v1 = hull.vertices[v_idx - 1]
            v2 = hull.vertices[v_idx]
            start = original[v1]
            end = original[v2]
            temp = point_to_line(enc, start, end)
            dist_list.append(temp[0])
        inside = point_in_poly(enc[0], enc[1], original[hull.vertices])
        if inside == True:
            dist_temp = -1. * min(dist_list)
        else:
            dist_temp = min(dist_list)
        point_dist.append(dist_temp)
    return np.array(point_dist)

birds_of_prey_dict, filenames_dict = get_concept_encodings(get_label_from_word('bird_of_prey'), 'model/vae2', 'train', stacked=False, include_filenames=True)
birds_of_prey_64_dict = get_concept_encodings(get_label_from_word('bird_of_prey'), 'model/vae64', 'train', stacked=False)
owl_label = get_label_from_word('great_grey_owl')
other_encodings = np.zeros((3600, 2))
other_encodings_64 = np.zeros((3600, 64))
other_filenames = []
i = 0
for label in birds_of_prey_dict.keys():
    if label == owl_label:
        owl_encodings = birds_of_prey_dict[label]
        owl_encodings_64 = birds_of_prey_64_dict[label]
        owl_filenames = [s.decode('utf-8') for s in filenames_dict[label]]
    else:
        other_encodings[1200*i:1200*(i+1)] = birds_of_prey_dict[label]
        other_encodings_64[1200*i:1200*(i+1)] = birds_of_prey_64_dict[label]
        other_filenames += [s.decode('utf-8') for s in filenames_dict[label]]
        i += 1
owl_encodings_dbscan = dbscan_filter(owl_encodings)

with open('model/vae2/normals.p', 'rb') as f:
    normals = pickle.load(f)
with open('model/vae64/normals.p', 'rb') as f:
    normals_64 = pickle.load(f)
owl_normal = normals[owl_label]
owl_normal_64 = normals_64[owl_label]

owl_hull = ConvexHull(owl_encodings)
owl_hull_dbscan = ConvexHull(owl_encodings_dbscan)

owl_dists = get_dists(owl_encodings, owl_encodings, owl_hull)
owl_dists_dbscan = get_dists(owl_encodings, owl_encodings_dbscan, owl_hull_dbscan)
other_dists = get_dists(other_encodings, owl_encodings, owl_hull)
other_dists_dbscan = get_dists(other_encodings, owl_encodings_dbscan, owl_hull_dbscan)

owl_results = {}
for i, (filename, enc, enc64, dist, dist_dbscan) in enumerate(list(zip(owl_filenames, owl_encodings, owl_encodings_64, owl_dists, owl_dists_dbscan))):
    p2 = owl_normal.pdf(enc) / owl_normal.pdf(owl_normal.mean)
    p64 = owl_normal_64.pdf(enc64) / owl_normal_64.pdf(owl_normal_64.mean)
    owl_results[filename] = {'enc': enc.astype(float).tolist(), 'dist': float(dist), 'dist_dbscan': float(dist_dbscan), 'p2': float(p2), 'p64': float(p64)}

other_results = {}
for i, (filename, enc, enc64, dist, dist_dbscan) in enumerate(list(zip(other_filenames, other_encodings, other_encodings_64, other_dists, other_dists_dbscan))):
    p2 = owl_normal.pdf(enc) / owl_normal.pdf(owl_normal.mean)
    p64 = owl_normal_64.pdf(enc64) / owl_normal_64.pdf(owl_normal_64.mean)
    other_results[filename] = {'enc': enc.astype(float).tolist(), 'dist': float(dist), 'dist_dbscan': float(dist_dbscan), 'p2': float(p2), 'p64': float(p64)}

with open('results/qualitative/owls.json', 'w+') as f:
    f.write(json.dumps(owl_results, indent=4))
with open('results/qualitative/owls.txt', 'w+') as f:
    f.write('filename\tdist\tdist_dbscan\tp2\tp64\n')
    for filename, res in owl_results.items():
        del res['enc']
        f.write('{0}\t{1}\n'.format(filename, '\t'.join([str(i) for i in res.values()])))
with open('results/qualitative/other.json', 'w+') as f:
    f.write(json.dumps(other_results, indent=4))
with open('results/qualitative/other.txt', 'w+') as f:
    f.write('filename\tdist\tdist_dbscan\tp2\tp64\n')
    for filename, res in other_results.items():
        del res['enc']
        f.write('{0}\t{1}\n'.format(filename, '\t'.join([str(i) for i in res.values()])))
