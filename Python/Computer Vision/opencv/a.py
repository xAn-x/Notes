import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import sklearn

import cv2
from utils import plot_images

img=cv2.imread("assets/open-cv.png",cv2.IMREAD_COLOR)
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thres=cv2.threshold(grey,127,255,cv2.THRESH_BINARY)

# cv2.findContours(image,mode,method)
contours, hierarchy=cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of Contours: {len(contours)}")

img_with_countours=cv2.drawContours(img.copy(),contours,-1,(0,255,255),3) 

plot_images([img,img_with_countours],1,2)