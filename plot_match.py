import cv2
import numpy as np
import os
from PIL import Image

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

pair_path = os.path.join('./hpatches_sequences/hpatches-sequences-release/v_bird/')

image1 = np.array(Image.open(os.path.join(pair_path, '1.ppm')))
image2 = np.array(Image.open(os.path.join(pair_path, '2.ppm')))

feat1 = np.load(os.path.join(pair_path+'1.ppm.cft'))
feat2 = np.load(os.path.join(pair_path+'2.ppm.cft'))


matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
print('Number of raw matches: %d.' % matches.shape[0])

keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
keypoints_right = feat2['keypoints'][matches[:, 1], : 2]


np.random.seed(0)
model, inliers = ransac(
    (keypoints_left, keypoints_right),
    ProjectiveTransform, min_samples=4,
    residual_threshold=3, max_trials=10
)
n_inliers = np.sum(inliers)


inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
all_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left]


inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
all_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right]


image1 = cv2.drawKeypoints(image1, [item for item in all_keypoints_left if item not in inlier_keypoints_left],\
                           image1, color=(0,0,255), flags=None)
image1 = cv2.drawKeypoints(image1, inlier_keypoints_left,image1, color=(0,255,0), flags=None)

image2 = cv2.drawKeypoints(image2, [item for item in all_keypoints_right if item not in inlier_keypoints_right],\
                           image2, color=(0,0,255), flags=None)
image2 = cv2.drawKeypoints(image2, inlier_keypoints_right,image2, color=(0,255,0), flags=None)


placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = np.zeros((50,50,3))
image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right,\
                         placeholder_matches, image3, matchColor=(0,255,0))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 0.7
fontColor              = (0,0,0)
thickness              = 2
lineType               = 2

cv2.putText(image3,'Inliers/Raw matches: '+ str(n_inliers)+'/'+str(matches.shape[0]),
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    thickness,
    lineType)

cv2.imshow('plot_match',image3)
cv2.waitKey(0)

