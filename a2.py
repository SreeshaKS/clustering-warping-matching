import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics
from itertools import cycle
import sys
from sklearn import cluster, metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics


def part1(image_list, n_clusters):
    lables = [ file.split('/')[-1] for file in image_list ]
    images = [ cv2.imread(file, 0) for file in image_list]

    descs_list = []
    for i in images:
        totalFeaturePoints = 1000
        orb = cv2.ORB_create(nfeatures=totalFeaturePoints)
        kp, descriptors = orb.detectAndCompute(i, None)
        descs_list.append(descriptors[0])

    def metric(x, y):
        x, y = np.float32(x), np.float32(y)
        matches =  cv2.BFMatcher().knnMatch(x, y, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m.distance)
        return sorted(good)[0]

    desc_distance = pairwise_distances(descs_list, descs_list, metric)

    affinityVal = 'precomputed'
    linkageVal = 'average'
    ac = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinityVal, linkage=linkageVal
        ).fit_predict(desc_distance)

    # print(metrics.silhouette_score(descs_list, ac, metric='euclidean'))
    final_str = ''
    for x in range(n_clusters):
        for points in np.where(ac==x):
            final_str += ' '.join([lables[p] for p in points]) + '\n'
    return final_str


if sys.argv[1] == 'part1':
    file_to_write = sys.argv[-1]
    n_clusters = int(sys.argv[2])
    image_list=[]
    image_files_wildcard = sys.argv[3]
    for x in range(3,len(sys.argv)-2):
        a = sys.argv[x]
        image_list.append(a)
    print('Clusters', n_clusters)
    print('output fie', file_to_write)
    st = part1(image_list, n_clusters)
    f = open(file_to_write,'w')
    f.write(st + " ")