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
  - 2000 features ( 83.3 % and 50 seconds TAT )
  - 1000 features - fastest results with a manageable dip in accuracy( 82.4 % 25 seconds TAT )


## Discussion
- ORB uses FAST and BRIEF
  - FAST compares the brightness of a sample pixel 'p' to the surrounding 16 pixels - selects keypoints which are darker than 'p'.
  - for each of these keypoints BRIEF picks two pixels at random from the keypoint's guassian smoothed surrounds
    - the first pixel ( kp1 ) is picked from surroundings smoothed by guassian kernel with SD of 'sigma1'
    - the second pixel ( kp2 ) is picked from surrounds smoothed by guassian kernel with SD of 'sigma2'
    - if kp2 is brigther than kp1 then one of the 128 bits is set to '1' otherwise '0'
  - BRIEF repeats this 128 times
- for images with higher average pixel intensity and similar features, performance tends to drop due to the above procedure of choosing darker or lighter pixels around keypoints.


### False Positives
![bigben-londoneye](https://github.iu.edu/cs-b657-sp2022/zseliger-sskuruva-idonbosc-a2/blob/main/fp-bigben-louvre.jpg)

- Here, the edges and the localities of the keypoints are quite similar. This increases the matching around keypoints since the gradients are similar caussing BRIEF to pick pixels around the key points

### False Negatives
![bigben-bigben](https://github.iu.edu/cs-b657-sp2022/zseliger-sskuruva-idonbosc-a2/blob/main/fn-bigben-bigben.jpg)

- Here, the edges and the localities around the keypoints have different intensities and hence the procedure fails since decriptor picked has different distances and fails the ratio test.

