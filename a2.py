#!/usr/bin/env python2

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics
from itertools import cycle
import sys

# method to generate SIFT features of images in a directory with some extension
def getImageSIFT(image_list):
    dictionarySize = 100 # image dictionary size is set to the desired value
    bow = cv2.BOWKMeansTrainer(dictionarySize)
    for file in image_list:
        i = cv2.imread(file, 0)
        if i is not None:
            sift = cv2.SIFT_create()
            _, desc = sift.detectAndCompute(i, None)
            bow.add(desc)
    return bow

# method to compute BOW Dictionary
def getBOWDictionary(BOW):
    dictionary = BOW.cluster()
    sift2 = cv2.SIFT_create()
    BOW_dict = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    BOW_dict.setVocabulary(dictionary)
    return BOW_dict

# method to compute BOW descriptors for images
def getImageBOW(image_list, BOWDiction):
    imgDesc = []
    img_array = []
    for file in image_list:
        i = cv2.imread(file, 0)
        if i is not None:
            sift = cv2.SIFT_create()
            des = BOWDiction.compute(i, sift.detect(i))
            imgDesc.append(des[0])
            img_array.append(file.split('/')[-1])
    return np.asarray(imgDesc), img_array


# method to perform hierarchical clustering; returns labels and centroids
def performHClustering(X, noCluster, affinityVal, linkageVal):
    print(noCluster)
    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=noCluster, affinity=affinityVal, linkage=linkageVal).fit_predict(X)
    #extracting centroids of hierarchical clusters
    codebook = []
    for i in range(ac.min(), ac.max()+1):
        codebook.append(X[ac == i].mean(0))
    centroid = np.vstack(codebook)
    return ac, centroid

# method that calls all the other methods
def part1(image_list, n_clusters):
    
    BOW = getImageSIFT(image_list)

    BOWDictionary = getBOWDictionary(BOW)

    desArray, img_array = getImageBOW(image_list, BOWDictionary)

    affinityVal = 'euclidean'
    linkageVal = 'average'
    labelH, centroidH = performHClustering(desArray, n_clusters, affinityVal, linkageVal)

    final_str = ''
    for x in range(n_clusters):
        for points in np.where(labelH==x):
            final_str += ' '.join([img_array[p] for p in points]) + '\n'
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
