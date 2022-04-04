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
from PIL import Image
import random

def metric(x, y):
    x, y = np.float32(x), np.float32(y)
    matches = cv2.BFMatcher().knnMatch(x, y, k=2)
    good = []
    good_dist = []

    distances = []
    
    for m, n in matches:  # for each descriptor, consider closest two matches
        distances.append(m.distance)
        if m.distance < 0.75 * n.distance:  # best match has to be 0.75 much closer than second best
            good_dist.append(m.distance+0.00001)
            good.append([m])


    if len(good_dist) == 0:
        return 0, good, []
    # return len(good), good # - this gives an accuracy of 81.4%
    return np.mean(good_dist) / sum(good_dist), good, good_dist # - this gives an accuracy of 82.4%

def prediction_accuracy(c_labels, lables, n, descs_list, images, kp_list):
    lables = np.array([i.split('_')[0]
                        for i in lables])  # retrieve names of files

    tp = 0
    tn = 0
    t = n * (n - 1)  # total pairs

    for i in range(len(c_labels)):
        for j in range(len(c_labels)):
            if i != j:  # do not compare the image with itself
                m, good, distances = metric(descs_list[i], descs_list[j])
                # out_img = np.zeros((200, 200, 3), np.uint8)
                # img3 = cv2.drawMatchesKnn(
                #     images[i], kp_list[i], images[j], kp_list[j], good, flags=2, outImg=out_img
                # )

                # If image belong to the same cluster and has the same filename
                if (c_labels[i] == c_labels[j]) & (lables[i] == lables[j]):
                    # fig, ax = plt.subplots()
                    # ax.hist(distances)
                    # ax.set_ylabel('count')
                    # ax.set_xlabel('distances')
                    # plt.savefig('results/tp-{}-{}.png'.format(lables[i], lables[j]))
                    # plt.close(fig)
                    tp += 1
                # If image belongs to different class and has a separate file name
                elif (c_labels[i] != c_labels[j]) & (lables[i] != lables[j]):
                    tn += 1
                # # FP
                # elif (c_labels[i] == c_labels[j]) & (lables[i] != lables[j]):
                #     fig, ax = plt.subplots()
                #     ax.hist(distances)
                #     ax.set_ylabel('count')
                #     ax.set_xlabel('distances')
                #     # plt.show()
                #     plt.savefig('results/fp-{}-{}.png'.format(lables[i], lables[j]))
                #     plt.close(fig)
                #     # cv2.imwrite('results/fp-{}-{}.jpg'.format(lables[i], lables[j]), img3)
                # # FN
                # elif (c_labels[i] != c_labels[j]) & (lables[i] == lables[j]):
                #     fig, ax = plt.subplots()
                #     ax.hist(distances)
                #     ax.set_ylabel('count')
                #     ax.set_xlabel('distances')
                #     plt.savefig('results/fn-{}-{}.png'.format(lables[i], lables[j]))
                #     plt.close(fig)
                    # cv2.imwrite('results/fn-{}-{}.jpg'.format(lables[i], lables[j]), img3)
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
            desc_distance[idx][idy], _, _ = metric(x, y)

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


#################### PART 2 ###########################

def apply_transformation(image_orig, matrix):
    ### https://www.codingame.com/playgrounds/2524/basic-image-manipulation/transformation ###
#   image_new = np.zeros((image_orig.width, image_orig.height))
    image_new = Image.new('RGB', image_orig.size, color = (0,0,0))

    for i in range(image_orig.width):
        for j in range(image_orig.height):
            loc = np.array([i,j,1])
            new_loc = np.dot(matrix, loc)
            final_loc=(new_loc[:-1]/new_loc[-1]).astype(int).tolist()
            pix_loc = image_orig.getpixel((i,j))

            if (final_loc[0]>0 and final_loc[0]<image_orig.width) and (final_loc[1]<image_orig.height and final_loc[1]>0):
                    image_new.putpixel((final_loc[0],final_loc[1]),pix_loc)

    image_new.save(sys.argv[5])    
    return image_new


def image_translation(image, s1, t1):
    x_value = s1[0] - t1[0]
    y_value = s1[1] - t1[1]
    trans_matrix = np.array([[1,0,x_value],[0,1,y_value],[0,0,1]])
    transformed_img = apply_transformation(image, trans_matrix)

def euclidian_image_transformation(image_orig, s1, t1, s2, t2):
#   image_new = np.zeros((image_orig.width, image_orig.height))
    image_new = Image.new('RGB', image_orig.size, color = (0,0,0))

    #Logic reference https://math.stackexchange.com/questions/3998354/construct-an-affine-transformation-given-the-image-of-2-points-without-skewing
    # and https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/MATH/homogenous.html 

    transformation_matrix_orig=np.array([[s1[0],-s1[1],1,0],
                  [s1[1],s1[0],0,1],
                  [s2[0],-s2[1],1,0],
                  [s2[1],s2[0],0,1]])

    transformation_matrix_dest=np.array([[t1[0],t1[1],t2[0],t2[1]]]).T

    final_transformation_matrix=np.linalg.solve(transformation_matrix_orig,transformation_matrix_dest)
    transformation_matrix=[[final_transformation_matrix[0,0],- final_transformation_matrix[1,0], final_transformation_matrix[2,0]],[ final_transformation_matrix[1,0], final_transformation_matrix[0,0],final_transformation_matrix[3,0]],[0,0,1]]

    new_img = apply_transformation(image_orig,transformation_matrix)

def affine_image_transformation(image_orig, s1, t1, s2, t2, s3, t3):  #shortened the name of the coordinates for ease - o - orig_coord and t - trans_coord
    
    ###Logic reference https://staff.fnwi.uva.nl/r.vandenboomgaard/IPCV20162017/LectureNotes/MATH/homogenous.html 

    A=[[s1[0],s1[1],1],[s2[0],s2[1],1],[s3[0],s3[1],1]]
    B=[[t1[0],t1[1],1],[t2[0],t2[1],1],[t3[0],t3[1],1]]

    transformation_matrix=np.linalg.solve(A,B).T

    new_img = apply_transformation(image_orig,transformation_matrix)

def projective_image_transformation(image_orig, s1, t1, s2, t2, s3, t3, s4, t4):

#Using logic from http://graphics.cs.cmu.edu/courses/15-463/2008_fall/Papers/proj.pdf

    projective=[[s1[0],s1[1],1,0,0,0,-(s1[0]*t1[0]),-(s1[1]*t1[0])],
                [s2[0],s2[1],1,0,0,0,-(s2[0]*t2[0]),-(s2[1]*t2[0])],
                [s3[0],s3[1],1,0,0,0,-(s3[0]*t3[0]),-(s3[1]*t3[0])],
                [s4[0],s4[1],1,0,0,0,-(s4[0]*t4[0]),-(s4[1]*t4[0])],
                [0,0,0,s1[0],s1[1],1,-(s1[0]*t1[1]),-(s1[1]*t1[1])],
                [0,0,0,s2[0],s2[1],1,-(s2[0]*t2[1]),-(s2[1]*t2[1])],
                [0,0,0,s3[0],s3[1],1,-(s3[0]*t3[1]),-(s3[1]*t3[1])],
                [0,0,0,s4[0],s4[1],1,-(s4[0]*t4[1]),-(s4[1]*t4[1])]]

    B = np.array([[t1[0],t2[0],t3[0],t4[0],t1[1],t2[1],t3[1],t4[1]]]).T

    final_transformation_matrix = np.linalg.solve(projective,B)
    final_transformation_matrix = np.append(final_transformation_matrix,[[1]],axis=0)
    transformation_matrix = np.reshape(final_transformation_matrix,(3,3))

    new_img = apply_transformation(image_orig,transformation_matrix)

def part2():
    n = int(sys.argv[2])

    image_file = sys.argv[3]

    image_orig = Image.open(image_file)

    if n==1:
        if len(sys.argv)<8:
            print("Incorrect number of arguments")
        s1 = eval(sys.argv[6])
        t1 = eval(sys.argv[7])
        image_translation(image_orig,s1,t1)
    elif n==2:
        if len(sys.argv)<10:
            print("Incorrect number of arguments")
        s1=eval(sys.argv[6])
        t1=eval(sys.argv[7])
        s2=eval(sys.argv[8])
        t2=eval(sys.argv[9])
        euclidian_image_transformation(image_orig,s1,t1,s2,t2)
    elif n==3:
        if len(sys.argv)<12:
            print("Incorrect number of arguments")
        s1=eval(sys.argv[6])
        t1=eval(sys.argv[7])
        s2=eval(sys.argv[8])
        t2=eval(sys.argv[9])
        s3=eval(sys.argv[10])
        t3=eval(sys.argv[11])
        affine_image_transformation(image_orig,s1,t1,s2,t2,s3,t3)
    elif n==4:
        if len(sys.argv)<14:
            print("Incorrect number of arguments")
        s1=eval(sys.argv[6])
        t1=eval(sys.argv[7])
        s2=eval(sys.argv[8])
        t2=eval(sys.argv[9])
        s3=eval(sys.argv[10])
        t3=eval(sys.argv[11])
        s4=eval(sys.argv[12])
        t4=eval(sys.argv[13])
        projective_image_transformation(image_orig,s1,t1,s2,t2,s3,t3,s4,t4)

# modified code from part 2 to handle input points as an array of matches
def make_hypothesis(samples):
    # Using logic from http://graphics.cs.cmu.edu/courses/15-463/2008_fall/Papers/proj.pdf
    projective=[[samples[0][0],samples[0][1],1,0,0,0,-(samples[0][0]*samples[0][2]),-(samples[0][1]*samples[0][2])],
                [samples[1][0],samples[1][1],1,0,0,0,-(samples[1][0]*samples[1][2]),-(samples[1][1]*samples[1][2])],
                [samples[2][0],samples[2][1],1,0,0,0,-(samples[2][0]*samples[2][2]),-(samples[2][1]*samples[2][2])],
                [samples[3][0],samples[3][1],1,0,0,0,-(samples[3][0]*samples[3][2]),-(samples[3][1]*samples[3][2])],
                [0,0,0,samples[0][0],samples[0][1],1,-(samples[0][0]*samples[0][3]),-(samples[0][1]*samples[0][3])],
                [0,0,0,samples[1][0],samples[1][1],1,-(samples[1][0]*samples[1][3]),-(samples[1][1]*samples[1][3])],
                [0,0,0,samples[2][0],samples[2][1],1,-(samples[2][0]*samples[2][3]),-(samples[2][1]*samples[2][3])],
                [0,0,0,samples[3][0],samples[3][1],1,-(samples[3][0]*samples[3][3]),-(samples[3][1]*samples[3][3])]]

    B = np.array([[samples[0][2],samples[1][2],samples[2][2],samples[3][2],samples[0][3],samples[1][3],samples[2][3],samples[3][3]]]).T

    final_transformation_matrix = np.linalg.solve(projective,B)
    final_transformation_matrix = np.append(final_transformation_matrix,[[1]],axis=0)
    transformation_matrix = np.reshape(final_transformation_matrix,(3,3))
    return transformation_matrix

def part3():
    img1 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
    output_filename = sys.argv[4]

    # get ORB descriptors
    orb = cv2.ORB_create(nfeatures=1000)
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

    # find matching points
    matches = []
    threshold_dist = 20
    for i in range(len(descriptors1)):
        min_dist = -1
        min_index = -1
        for j in range(len(descriptors2)):
            dist = cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
            if dist < min_dist or min_index == -1:
                min_dist = dist
                min_index = j
        
        if min_dist < threshold_dist and min_dist > 0:
            matches.append([int(keypoints1[i].pt[0]), int(keypoints1[i].pt[1]), int(keypoints2[min_index].pt[0]), int(keypoints2[min_index].pt[1])])

    # RANSAC, find projection to stitch images together
    hypotheses = []
    voting = []
    num_rounds = 1000
    threshold_hypothesis = 2
    for r in range(num_rounds):
        samples = random.sample(matches, 4)
        h = make_hypothesis(samples)

        # if hypothesis has been seen before, count as vote instead of new hypothesis
        hypothesis_new = True
        for i in range(len(hypotheses)):
            if abs(np.sum(hypotheses[i] - h)) < threshold_hypothesis:
                voting[i] += 1
                hypothesis_new = False

        # new hypothesis
        if hypothesis_new:
            hypotheses.append(h)
            voting.append(1)

    # get hypothesis with most votes and make 3x3 matrix
    max_index = 0
    for i in range(len(hypotheses)):
        if voting[i] > voting[max_index]:
            max_index = i
    final_proj = hypotheses[max_index]

    # transform 2nd image to match projection in image 1
    img1 = Image.open(sys.argv[2])
    img2 = Image.open(sys.argv[3])

    # make new images and pull from both img1 and img2
    new_width = max(img1.width, img2.width)
    new_height = max(img1.height, img2.height)
    final_img = Image.new('RGB', (new_width, new_height))
    # transform 2nd image coordinates to match projection in image 1
    for x in range(final_img.width):
        for y in range(final_img.height):
            # transform into 2nd image coordinates
            # grab the pixel at that spot in the 2nd image
            new_coords = np.dot(final_proj, [x, y, 1])
            new_x = int(new_coords[0] / new_coords[2])
            new_y = int(new_coords[1] / new_coords[2])
            new_p = (0,0,0)
            if new_x > 0 and new_x < img2.width and new_y > 0 and new_y < img2.height:
                new_p = img2.getpixel((new_x, new_y))
            final_img.putpixel((x, y), new_p)

    # combine with pixel in first, original image
    for x in range(img1.width):
        for y in range(img1.height):
            if final_img.getpixel((x,y)) == (0,0,0):
                final_img.putpixel((x,y), img1.getpixel((x,y)))
            else:
                old_p = final_img.getpixel((x,y))
                im1_p = img1.getpixel((x,y))
                new_p = (int((old_p[0]+im1_p[0])/2), int((old_p[1]+im1_p[1])/2), int((old_p[2]+im1_p[2])/2))
                final_img.putpixel((x,y), new_p)
    final_img.save(output_filename)

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
    
elif sys.argv[1]=='part2':
    part2()

elif sys.argv[1]=='part3':
    part3()