import json
from scipy.spatial.distance import euclidean, cosine
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

with open('results/qualitative/owls.json') as f:
    owls = json.load(f)
with open('results/qualitative/other.json') as f:
    other = json.load(f)

encodings = []
best_boundary = None
best_inside = None
best_outside = None
h = 1

for i, (filename, owl) in enumerate(list(owls.items())):
    if owl['dist_dbscan'] == 0:
        encodings.append(owl['enc'])
        nearest_inside = None
        nearest_outside = None
        min_dist_inside = 1
        min_dist_outside = 1
        for f2, o2 in owls.items():
            dist = euclidean(owl['enc'], o2['enc'])
            cos = cosine(owl['enc'], o2['enc'])
            if o2['dist_dbscan'] < 0 and dist < min_dist_inside and cos < .0005 and dist > .01:
                min_dist_inside = dist
                nearest_inside = f2, o2
        for f2, o2 in other.items():
            dist = euclidean(owl['enc'], o2['enc'])
            cos = cosine(owl['enc'], o2['enc'])
            if o2['dist_dbscan'] > 0 and dist < min_dist_outside and cos < .0005 and dist > .01:
                min_dist_outside = dist
                nearest_outside = f2, o2
        if not nearest_inside == None and not nearest_outside == None:
            oh = min_dist_inside + min_dist_outside
            if oh < h:
                h = oh
                best_boundary = filename, owl
                best_inside = nearest_inside
                best_outside = nearest_outside

'''for i, (filename, owl) in enumerate(list(owls.items())):
    if owl['dist'] == 0:
        encodings.append(owl['enc'])
        nearest_inside = None
        nearest_outside = None
        min_dist_inside = 1
        min_dist_outside = 1
        for f2, o2 in owls.items():
            dist = euclidean(owl['enc'], o2['enc'])
            cos = cosine(owl['enc'], o2['enc'])
            if o2['dist'] < 0 and dist < min_dist_inside and cos < .001 and dist > .05:
                min_dist_inside = dist
                nearest_inside = f2, o2
        for f2, o2 in other.items():
            dist = euclidean(owl['enc'], o2['enc'])
            cos = cosine(owl['enc'], o2['enc'])
            if o2['dist'] > 0 and dist < min_dist_outside and cos < .001 and dist > .05:
                min_dist_outside = dist
                nearest_outside = f2, o2
        if not nearest_inside == None and not nearest_outside == None:
            oh = min_dist_inside + min_dist_outside
            if oh < h:
                h = oh
                best_boundary = filename, owl
                best_inside = nearest_inside
                best_outside = nearest_outside'''

encodings = np.array(encodings)
hull = ConvexHull(encodings)
x = np.array([best_boundary[1]['enc'], best_inside[1]['enc'], best_outside[1]['enc']])

print(best_boundary)
print(best_inside)
print(best_outside)

plt.scatter(x[:,0], x[:, 1])
for simplex in hull.simplices:
    plt.plot(encodings[simplex, 0], encodings[simplex, 1], 'k-')
plt.show()
