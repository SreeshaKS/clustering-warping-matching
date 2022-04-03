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


def metric(x, y):
    x, y = np.float32(x), np.float32(y)
    matches = cv2.BFMatcher().knnMatch(x, y, k=2)
    good = []
    good_dist = []
    for m, n in matches:  # for each descriptor, consider closest two matches
        if m.distance < 0.75 * n.distance:  # best match has to be 0.75 much closer than second best
            good_dist.append(m.distance+0.00001)
            good.append([m])

    if len(good_dist) == 0:
        return 0, good
    # return len(good), good # - this gives an accuracy of 81.4%
    return np.mean(good_dist) / sum(good_dist), good # - this gives an accuracy of 82.4%

def prediction_accuracy(c_labels, lables, n, descs_list, images, kp_list):
    lables = np.array([i.split('_')[0]
                        for i in lables])  # retrieve names of files

    tp = 0
    tn = 0
    t = n * (n - 1)  # total pairs

    for i in range(len(c_labels)):
        for j in range(len(c_labels)):
            if i != j:  # do not compare the image with itself
                m, good = metric(descs_list[i], descs_list[j])
                out_img = np.zeros((200, 200, 3), np.uint8)
                img3 = cv2.drawMatchesKnn(
                    images[i], kp_list[i], images[j], kp_list[j], good, flags=2, outImg=out_img
                )
                # If image belong to the same cluster and has the same filename
                if (c_labels[i] == c_labels[j]) & (lables[i] == lables[j]):
                    tp += 1
                # If image belongs to different class and has a separate file name
                elif (c_labels[i] != c_labels[j]) & (lables[i] != lables[j]):
                    tn += 1
                # FP
                # elif (c_labels[i] == c_labels[j]) & (lables[i] != lables[j]):
                #     cv2.imwrite('results/fp-{}-{}.jpg'.format(lables[i], lables[j]), img3)
                # FN
                # elif (c_labels[i] != c_labels[j]) & (lables[i] == lables[j]):
                #     cv2.imwrite('results/fn-{}-{}.jpg'.format(lables[i], lables[j]), img3)

    return ((tp+tn)/t) * 100

def part1(image_list, n_clusters):
    lables = [file.split('/')[-1] for file in image_list]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_list]
    n_images = len(images)

    descs_list = []
    kp_list = []
    
    # read images and compute orb descriptors
    for i in images:
        totalFeaturePoints = 1000
        orb = cv2.ORB_create(nfeatures=totalFeaturePoints)
        kp, descriptors = orb.detectAndCompute(i, None)
        kp_list.append(kp)
        descs_list.append(descriptors)

    # Compute the distance matrix
    desc_distance = np.zeros(shape=(n_images, n_images))
    for idx, x in enumerate(descs_list):
        for idy, y in enumerate(descs_list):
            desc_distance[idx][idy], _ = metric(x, y)

    affinityVal = 'precomputed'
    # complete gives the highest accuracy of 83 among ['complete', 'average', 'single']
    linkageVal = 'complete'

    ac = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkageVal, affinity=affinityVal
    ).fit(desc_distance)

    lines = [''] * len(ac.labels_)
    for idx, p in enumerate(ac.labels_):
        lines[p] += lables[idx] + ' '

    return '\n'.join(lines), prediction_accuracy(
        ac.labels_, lables, n_images, descs_list, images, kp_list
        )


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
