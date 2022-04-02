import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics
from itertools import cycle
import sys
from sklearn import cluster, metrics


def part1(image_list, n_clusters):
    lables = [ file.split('/')[-1] for file in image_list ]
    images = [ cv2.imread(file, 0) for file in image_list]

    descs_list = []
    for i in images:
        totalFeaturePoints = 1000
        orb = cv2.ORB_create(nfeatures=totalFeaturePoints)
        kp, descriptors = orb.detectAndCompute(i, None)
        descs_list.append(descriptors[0])

    affinityVal = 'euclidean'
    linkageVal = 'average'
    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinityVal, linkage=linkageVal).fit_predict(descs_list)

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