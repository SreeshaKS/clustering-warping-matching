# a2 - Part 1 - Image clustering using ORB

## General Procedure
- Create descritors using ORB
- Run sklearn.Aglomeratice clustering algorithm with the number of given clusters
- used pairwise_distance function to calculate distance matrix
- used the BFMatcher.knnMatch with k=2 and a ratio test with 0.75 threshold
- metric used was merely the total number of matches
- by using mean(distance of matches) / (total distance of all matches) - accuracy increases by 1 % from 81 - 82

## A few observations
- increasing the features from 1000 ( 81 % ) to 2000 ( 82 % ), increases accuracy 1%
- using 'complete' linkage gives the higest accuracy and 'single' gives the lowest
- Accuracy increases to 83.3 % from 81 %
  - using complete linkage
  - 'mean(distance of matches) / (total distance of all matches)'
  - 2000 features ( 50 seconds TAT )
  - 1000 features - gives the fastest results with a manageable dip in accuracy to 82.4 % ( 25 seconds TAT )

### False Positives

### False Negatives