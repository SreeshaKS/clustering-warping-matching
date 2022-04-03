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
import sklearn


def part1(image_list, n_clusters):
    lables = [file.split('/')[-1] for file in image_list]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_list]
    n_images = len(images)

    descs_list = []
    for i in images:
        totalFeaturePoints = 1000
        orb = cv2.ORB_create(nfeatures=totalFeaturePoints)
        kp, descriptors = orb.detectAndCompute(i, None)
        descs_list.append(descriptors)

    def metric(x, y):
        x, y = np.float32(x), np.float32(y)
        matches = cv2.BFMatcher().knnMatch(x, y, k=2)
        good = []
        for m, n in matches:  # for each descriptor, consider closest two matches
            if m.distance < 0.75 * n.distance:  # best match has to be 0.75 much closer than second best
                good.append(m.distance)
        s = sorted(good)

        if len(good) == 0:
            return 0
        return np.mean(good) / len(good)

    def prediction_accuracy(c_labels, lables, n):
        lables = np.array([i.split('_')[0] for i in lables])  # retrieve names of files

        tp = 0
        tn = 0
        t = n*(n-1)  # total pairs

        for i in range(len(c_labels)):
            for j in range(len(c_labels)):
                if i != j:  # do not compare the image with itself
                    # If image belong to the same cluster and has the same filename
                    if (c_labels[i] == c_labels[j]) & (lables[i] == lables[j]):
                        tp += 1
                    # If image belongs to different class and has a separate file name
                    elif (c_labels[i] != c_labels[j]) & (lables[i] != lables[j]):
                        tn += 1

        return ((tp+tn)/t) * 100

    # Compute the distance matrix
    desc_distance = np.zeros(shape=(n_images, n_images))
    for idx, x in enumerate(descs_list):
        for idy, y in enumerate(descs_list):
            desc_distance[idx][idy] = metric(x, y)

    # desc_distance = pairwise_distances(descs_list, descs_list, metric)

    affinityVal = 'precomputed'
    linkageVal = 'complete' # complete gives the highest accuracy of 83 among ['complete', 'average', 'single']

    ac = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkageVal, affinity=affinityVal
    ).fit(desc_distance)

    lines = [''] * len(ac.labels_)
    for idx, p in enumerate(ac.labels_):
        lines[p] += lables[idx] + ' '

    return '\n'.join(lines), prediction_accuracy(ac.labels_, lables, n_images)








if sys.argv[1] == 'part1':
    file_to_write = sys.argv[-1]
    n_clusters = int(sys.argv[2])
    image_list = []
    image_files_wildcard = sys.argv[3]
    for x in range(3, len(sys.argv)-2):
        a = sys.argv[x]
        image_list.append(a)
    print('Clusters - ', n_clusters)
    print('Output file - ', file_to_write)

    st, accuracy = part1(image_list, n_clusters)
    print('The Accuracy is {}.'.format(accuracy))

    f = open(file_to_write, 'w')
    f.write(st + " ")
