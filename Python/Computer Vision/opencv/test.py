import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import sklearn

import cv2
from utils import plot_images

blob_img=cv2.imread("assets/BlobTest.png",cv2.IMREAD_GRAYSCALE)


params=cv2.SimpleBlobDetector_Params()

# change threshold
params.minThreshold=10
params.maxThreshold=200

# area
params.filterByArea=True
# params.minArea=100
# params.maxArea=1000

# Filter by Cirularity
params.filterByCircularity=True
params.minCircularity=0.1

# Convexity
params.filterByConvexity=True
params.minConvexity=0.81

# Filter by Inertia
params.filterByInertia=True
params.minInertiaRatio=0.01

detector=cv2.SimpleBlobDetector_create(params)

keypoints=detector.detect(blob_img)

# drawKeyPoints(src_img,keypoints,dest_img,color,flags)
img_with_keypoints=cv2.drawKeypoints(blob_img,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plot_images([blob_img,img_with_keypoints],1,2,["original","keypoints"])