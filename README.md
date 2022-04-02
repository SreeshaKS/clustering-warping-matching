# a2

## General Procedure
- Create descritors using ORB
- Run sklearn.Aglomeratice clustering algorithm with the number of given clusters
- used pairwise_distance function to calculate distance matrix
- used the BFMatcher.knnMatch with k=2 and a ratio test with 0.75 threshold
- sorted the matches which pass the threshold and picked the worst distance descriptor as a metric for distance matrix