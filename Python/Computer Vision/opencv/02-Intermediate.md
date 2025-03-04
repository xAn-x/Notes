## Object detection using HSV:

![[Pasted image 20250110092100.png]]

Let's say we  want to detect green colour ball in the below image

![[Pasted image 20250110092210.png]]

```python
import cv2
import numpy as np

cv2.namedWindow("tracking")

# These trackbar help us to adjust hsv values
cv2.CreateTrackbar("lh","tracking",0,255,lambda x:None)
cv2.CreateTrackbar("ls","tracking",0,255,lambda x:None)
cv2.CreateTrackbar("lv","tracking",0,255,lambda x:None)


cv2.CreateTrackbar("uh","tracking",0,255,lambda x:None)
cv2.CreateTrackbar("us","tracking",0,255,lambda x:None)
cv2.CreateTrackbar("uv","tracking",0,255,lambda x:None)

while True:
	frame=cv2.imshow('balls.png')
	hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	l_h=cv2.getTrackbarPos("lh","tracking")	
	l_s=cv2.getTrackbarPos("ls","tracking")	
	l_v=cv2.getTrackbarPos("lv","tracking")	

	u_h=cv2.getTrackbarPos("uh","tracking")	
	u_s=cv2.getTrackbarPos("us","tracking")	
	u_v=cv2.getTrackbarPos("uv","tracking")	

	lowerbound=np.array((l_h,l_s,l_v))
	upperbound=np.array((u_h,u_s,u_v))
	
	mask=cv2.inRange(hsv_frame,lowerbound,upperbound)
	
	res=cv2.bitwise_and(frame,frame,mask=mask)
	
	cv2.imshow("Detections",rest)
	if cv2.waitkey(1)==ord('q'):
		break

cv2.destroyAllWindows()
```

## Thresholding using OpenCV

Thresholding in OpenCV is an image segmentation technique used to create a binary image from a grayscale or color image.  It works by setting all pixels above or below a certain threshold value to a specific value (usually white or black).

**How it Works:**

```python
new_pixel=v1 if old_pixel>threshold else v2
```

**Types of Thresholding in OpenCV:**

OpenCV provides several thresholding types via the `cv2.threshold()` function:

* **`cv2.THRESH_BINARY`:**  Simple thresholding.  Pixels above the threshold become white; those below become black.
* **`cv2.THRESH_BINARY_INV`:** Inverted binary thresholding. Pixels above the threshold become black; those below become white.
* **`cv2.THRESH_TRUNC`:**  Truncates pixel values above the threshold to the threshold value.  Pixels below remain unchanged.
* **`cv2.THRESH_TOZERO`:** Sets pixels below the threshold to zero. Pixels above remain unchanged.
* **`cv2.THRESH_TOZERO_INV`:** Sets pixels above the threshold to zero. Pixels below remain unchanged.
* **`cv2.THRESH_OTSU`:** Otsu's method automatically calculates an optimal threshold value based on the image histogram. Often used with `cv2.THRESH_BINARY` or `cv2.THRESH_BINARY_INV`.
* **`cv2.THRESH_TRIANGLE`:** Triangle algorithm, another automatic thresholding method.


**Uses of Thresholding:**

* **Image Segmentation:** Separating objects from the background.
* **Object Detection:**  A pre-processing step to simplify image analysis.
* **Edge Detection:**  Combined with edge detectors like Canny to highlight edges.
* **Optical Character Recognition (OCR):**  Creating binary images of text for easier character recognition.
* **Shape Analysis:**  Analysing the shapes of objects in binary images.


**Example:**

```python
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE) # Load grayscale image

# Binary thresholding
# cv2.threshold(img,threshold,maxval,thres-type)
thresh_val, thresh_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
thresh_val, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Original', img)
cv2.imshow('Binary Thresholding', thresh_binary)
cv2.imshow('Otsu Thresholding', thresh_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
## Adaptive Thresholding

Adaptive thresholding in OpenCV is a more advanced thresholding technique that doesn't rely on a single global threshold value for the entire image. Instead, it calculates a threshold value for each small region (neighborhood) within the image. This makes it particularly effective for images with uneven lighting or varying background intensities.

**How it Works:**

1. **Neighborhood:**  A neighborhood (e.g., a rectangular area) around each pixel is considered.
2. **Threshold Calculation:**  A threshold value is calculated for that neighborhood based on a chosen method. Common methods include:
    * **`cv2.ADAPTIVE_THRESH_MEAN_C`:** The threshold is the mean of the neighborhood pixel values minus a constant `C`.
    * **`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`:** The threshold is a weighted sum (Gaussian-weighted) of the neighborhood values minus a constant `C`.

3. **Thresholding:**  The pixel in question is then compared to the calculated threshold for its neighborhood.  If the pixel value is above the threshold, it's set to the maximum value (usually 255, white); otherwise, it's set to 0 (black).

**`cv2.adaptiveThreshold()` Function:**

```python

dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

* `src`: Grayscale input image.
* `maxValue`: Maximum value assigned to pixels above the threshold.
* `adaptiveMethod`:  `cv2.ADAPTIVE_THRESH_MEAN_C` or `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`.
* `thresholdType`:  Thresholding type (usually `cv2.THRESH_BINARY` or `cv2.THRESH_BINARY_INV`).
* `blockSize`: Size of the neighborhood area (must be odd).
* `C`: Constant subtracted from the mean or weighted mean.

**Use Cases:**

* **Document Scanning/OCR:**  Handles uneven lighting conditions common in scanned documents, making text extraction cleaner.
* **Medical Imaging:**  Segments features in images like X-rays or MRIs where intensity variations can be significant.
* **Object Detection in Varying Lighting:**  Improves object detection when lighting conditions are not uniform across the image.
* **Number Plate Recognition:**  Extracts characters from license plates with varying backgrounds and lighting.


**Example:**

```python
import cv2

img = cv2.imread('uneven_lighting.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale image

# Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Original', img)
cv2.imshow('Adaptive Mean', adaptive_mean)
cv2.imshow('Adaptive Gaussian', adaptive_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
## Morphological Transformation

Morphological transformations are operations based on the shape of an image, **typically performed on binary images**, but also applicable to grayscale images. they use a structuring element (kernel) to probe and change the shape of image features.

a kernel tells you how to change the value of any given pixel by combining it with different amounts of neighbouring pixels.

### Common morphological operations:

**erosion**: shrinks white regions.  removes small objects/noise. Use for thinning them edges
$dst(x,y) = \min_{(x',y') \in kernel} src(x+x', y+y')$

**dilation**: expands white regions.  connects broken parts/fills holes. Use for thick edges.
$dst(x,y) = \max_{(x',y') \in kernel} src(x+x', y+y')$

**opening**: erosion then dilation.  removes noise while preserving shape.

**closing**: dilation then erosion. closes small holes, joins objects.

**gradient**: difference between dilation and erosion. detects edges.

**top hat**: input image minus its opening.  finds small bright spots.

**black hat**: closing minus input image. finds small dark spots.


```python
import cmath
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def  plot_images(images:list[np.ndarray],names:list[str],shape:tuple[int,int]):
    assert len(images)==math.prod(shape),"number of images does not match shape"
    fig,axes=plt.subplots(nrows=shape[0],ncols=shape[1],figsize=(10,10))
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i],cmap="gray")
        ax.set_title(names[i])
        ax.axis("off")
        ax.set_aspect("equal")
        
    plt.tight_layout()


img=cv2.imread("balls.png",cv2.imread_grayscale)
_,mask=cv2.threshold(img,220,255,cv2.thresh_binary_inv)

kernal=np.ones((3,3),np.uint8)

dilate=cv2.dilate(mask,kernal,iterations=1)
erode=cv2.erode(mask,kernal,iterations=1)

opening=cv2.morphologyex(mask,cv2.morph_open,kernal)
closing=cv2.morphologyex(mask,cv2.morph_close,kernal)

plot_images([img,mask,dilate,erode,opening,closing],["original","mask","dilate","erode","opening","closing"],(2,3))

```
![[pasted image 20250110135416.png]]


### Blob Detection

A Blob is a group of connected pixels in an image that share some common property ( E.g, grayscale value )

![[BlobTest.jpg]]

```python
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
```


![[Pasted image 20250126215421.png]]

### Smoothening in opencv

 smoothening in opencv refers to techniques used to reduce noise and detail in an image, resulting in a "smoother" appearance.  several methods can achieve this:

* **averaging:**  replaces each pixel with the average of its neighbors.  this is done by convolving the image with a normalized box filter (kernel).  effective for reducing random noise but can blur edges.  In OpenCV, `cv2.blur()` or `cv2.boxFilter()` can be used.

* **Gaussian Blurring:**  Similar to averaging, but uses a weighted average where closer pixels contribute more.  Provides better edge preservation than simple averaging.  `cv2.GaussianBlur()` is used in OpenCV.  Requires kernel size and standard deviation (sigma) as parameters.  A larger sigma increases the blur.

* **Median Blurring:** Replaces each pixel with the median value of its neighbors. Particularly effective at removing salt-and-pepper noise while preserving sharp edges. `cv2.medianBlur()` is used, requiring only kernel size.

* **Bilateral Filtering:**  More sophisticated; smooths while preserving edges by considering both spatial distance and intensity difference.  Slower than other methods but provides high-quality smoothing.  `cv2.bilateralFilter()` requires parameters for diameter, sigmaColor (intensity differences), and sigmaSpace (spatial differences).

```python
# load image
from curses import KEY_RESIZE

from numpy import median


img=cv2.imread("test.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

kernel=np.ones((5,5),np.float32)/20

# apply convolution
conv2d=cv2.filter2D(img,ddepth=-1,kernel=kernel)

# blur image
blur=cv2.blur(img,(5,5))

# gaussian blur: more blur than normal blur, apply different weights to the pixels in both x and y directions, designed specifically for removing high frequency noise
guassian_blur=cv2.GaussianBlur(img,(5,5),0)

# median blur: replaces each pixel with the median of its neighboring pixels, effective in removing salt and pepper noise
median_blur=cv2.medianBlur(img,5) # kernel size must be odd

# bilateral blur: preserves edges while blurring the image, effective in removing noise while preserving edges
bilateral_blur=cv2.bilateralFilter(img,9,75,75)

# plot images
plot_images([img,conv2d,blur,guassian_blur,median_blur,bilateral_blur],2,3,["Original","Convolution","Blur","Gaussian Blur","Median Blur","Bilateral Blur"])
```
![[Pasted image 20250116204558.png]]

### Image Gradients & Edge Detection

Edge detection is an image-processing technique that is used to identify the boundaries (edges) of objects or regions within an image.**Sudden changes in pixel intensity characterize edges.**

Image gradients are directional changes in the intensity or color of an image.  They are fundamental for edge detection, as edges represent sharp changes in intensity.  OpenCV provides several methods for calculating image gradients:

* **Sobel Operator:**  Calculates the gradient in the x and y directions using separate kernels.  It emphasizes edges in a specific direction.  `cv2.Sobel()` is used, requiring parameters for the image, depth, and x/y order (1, 0 for x gradient, 0, 1 for y gradient).  Kernel size can also be specified.

* **Scharr Operator:**  Similar to Sobel but uses a slightly different kernel optimized for rotational invariance.  Provides better results than Sobel for small kernels (3x3).  `cv2.Scharr()` is used similarly to `cv2.Sobel()`.

* **Laplacian Operator:**  Calculates the second-order derivative, representing regions of rapid intensity change.  Useful for detecting edges and corners. `cv2.Laplacian()` is used.

After calculating the x and y gradients (e.g., using Sobel or Scharr), the magnitude and direction of the gradient can be computed:

* **Gradient Magnitude:** Represents the strength of the edge.  Calculated as $\sqrt{G_x^2 + G_y^2}$, where $G_x$ and $G_y$ are the x and y gradients, respectively.  `cv2.magnitude()` can be used.

* **Gradient Direction:** Represents the orientation of the edge. Calculated as $\arctan(\frac{G_y}{G_x})$.  `cv2.phase()` can be used.


```python
## Image Gradient and Edge Detection

img=cv2.imread("tesla.png",cv2.IMREAD_GRAYSCALE)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# laplacian filter: high pass filter, used to find edges in an image, 
lap=cv2.Laplacian(img,ddepth=2,ksize=1) 

# sobel filter: high pass filter, used to find edges in an image, more sensitive to noise than laplacian filter
sudoku=cv2.imread("sudoku.png",cv2.IMREAD_GRAYSCALE)

sobelX=cv2.Sobel(sudoku,ddepth=-1,dx=1,dy=0,ksize=3) 
sobelY=cv2.Sobel(sudoku,ddepth=-1,dx=0,dy=1,ksize=3)
sobelXY=sobelX | sobelY

plot_images([img,lap,sudoku,sobelX,sobelY,sobelXY],3,2,["Original","Laplacian","sudoku","sobelX","sobelY","Sobel Combine"])
```

![[Pasted image 20250116211805.png]]

### Canny Edge Detection

Canny Edge Detection is a multi-stage algorithm used to detect edges in images while minimizing noise and accurately capturing true edges.  Here's a breakdown of the steps:

1. **Noise Reduction:**  The image is smoothed using a Gaussian filter to reduce noise that could be misinterpreted as edges.  The size of the Gaussian kernel affects the amount of smoothing.

2. **Gradient Calculation:**  The intensity gradients of the smoothed image are calculated.  Typically, the Sobel operator is used to find the gradients in the x and y directions ($G_x$ and $G_y$).  From these, the gradient magnitude ($G = \sqrt{G_x^2 + G_y^2}$) and direction ($\theta = \arctan(\frac{G_y}{G_x})$) are computed.

3. **Non-Maximum Suppression:**  This step thins the edges by ensuring only the local maxima of gradient magnitudes are retained.  For each pixel, its gradient magnitude is compared to the magnitudes of its neighbors along the gradient direction.  If the pixel's magnitude is not the largest, it is suppressed (set to zero).

4. **Double Thresholding:**  Two thresholds, a high threshold and a low threshold, are used to classify edge pixels.  Pixels with gradient magnitudes above the high threshold are considered strong edge pixels.  Pixels below the low threshold are discarded.  Pixels between the two thresholds are considered weak edge pixels.

5. **Edge Tracking by Hysteresis:**  This final step connects weak edge pixels to strong edge pixels if they are adjacent.  The idea is that weak edges connected to strong edges are likely part of a true edge.  Weak edges not connected to strong edges are discarded.  This hysteresis process helps to create continuous and connected edges.

```python
## Canny Edge Detection
img=cv2.imread("open-cv.png",cv2.IMREAD_GRAYSCALE)
canny=cv2.Canny(img,100,200)
plot_images([img,canny],1,2,["Original","Canny"])
```
![[Pasted image 20250116212319.png]]

