### Image Pyramids in OpenCV
Way to create a set of images with different resolutions, all derived from the same original image.

There are two main types:

- **Gaussian Pyramids:** These are formed by repeatedly blurring and down-sampling the image. This effectively reduces the image size while smoothing out details. *Useful for things like image blending and finding larger features.* In OpenCV, you can use `pyrDown()` to create a smaller image and `pyrUp()` to upscale (though upscaling doesn't recover lost detail, it just makes the image bigger).


```python
import cv2

img = cv2.imread("my_image.jpg")
smaller_img = cv2.pyrDown(img) # Downsample->0.5x
larger_img = cv2.pyrUp(smaller_img) # Upsample->2x
```

![[Pasted image 20250118212113.png]]

- **Laplacian Pyramids:** These store the _difference_ between two consecutive levels in a Gaussian pyramid. They're used for image *reconstruction* and are *calculated by subtracting the up-sampled version of a Gaussian pyramid level from the original level.* This highlights the details lost during down-sampling. 
  
  While there's no direct `pyrLaplacian()` function, they are constructed using `pyrDown()` and `pyrUp()` as intermediate steps. Their main utility is in reconstructing the original image from the pyramid representation. 
  

```python
import cv2
from utils import get_pixel_info,plot_images

img=cv2.imread("assets/tesla.png",0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


img=cv2.resize(img,(512,512)) # make sure them dim are even else cause bradcasting issue

guassian_pyramid=[img]
for i in range(3):
    img=cv2.pyrDown(img)
    guassian_pyramid.append(img)


laplacian_pyramid=[guassian_pyramid[-1]]
for i in range(len(guassian_pyramid)-1,0,-1):
    gaussian_expanded=cv2.pyrUp(guassian_pyramid[i]) # upsample then sub so they have same size
    laplacian_pyramid.append(cv2.subtract(gaussian_expanded,guassian_pyramid[i-1])) 

plot_images(laplacian_pyramid,1,5,figsize=(12,12)) # More like edge detection
```
  
  ![[Pasted image 20250118215349.png]]

## Mixing and Blending Images using Image Pyramids

Image blending with pyramids basically involves creating pyramids for each image, combining them, and then reconstructing a blended image. Here's a simplified breakdown:

1. **Compute Gaussian Pyramids:** *Create Gaussian pyramids for both images* you want to blend.  This gives you sets of images at different scales.

2. **Compute Laplacian Pyramids:** From the Gaussian pyramids, *calculate the Laplacian pyramids for each image*.  These pyramids contain the details lost at each level of down-sampling.

3. **Join Pyramids:**  *Combine the Laplacian pyramids level by level.* You might, for example, take the left half of one image's Laplacian pyramid level and the right half of the other's, joining them to create a combined level.

4. **Reconstruct Blended Image:**  Starting from the smallest level of the combined Laplacian pyramid, *upscale each level and add it to the next larger level.*  This reconstructs the final, blended image.

```python
import re
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import sklearn

import cv2
from utils import generate_k_pyramid,generte_laplacianPyr_from_gaussianPyr,PYRAMID_DIRN, plot_images

apple=cv2.imread("assets/apple.png",cv2.IMREAD_COLOR)
orange=cv2.imread("assets/orange.png",cv2.IMREAD_COLOR)

apple=cv2.resize(apple,(512,512))
orange=cv2.resize(orange,(512,512))

apple_pyr=generate_k_pyramid(apple,3,PYRAMID_DIRN.Down)
orange_pyr=generate_k_pyramid(orange,3,PYRAMID_DIRN.Down)

apple_laplacian_pyr=generte_laplacianPyr_from_gaussianPyr(apple_pyr)
orange_laplacian_pyr=generte_laplacianPyr_from_gaussianPyr(orange_pyr)

# Join left half of apple to right half of orange in each layer
blends=[]
n=len(apple_laplacian_pyr)
for apple_pyr,orange_pyr in zip(apple_laplacian_pyr,orange_laplacian_pyr):
    blend=np.hstack((apple_pyr[:,0:apple_pyr.shape[1]//2],orange_pyr[:,apple_pyr.shape[1]//2:]))
    blends.append(blend)

reconstructed_img=blends[0]
for i in range(1,n):
    reconstructed_img=cv2.pyrUp(reconstructed_img)
    reconstructed_img=cv2.add(reconstructed_img,blends[i])

apple=cv2.cvtColor(apple,cv2.COLOR_BGR2RGB)
orange=cv2.cvtColor(orange,cv2.COLOR_BGR2RGB)
reconstructed_img=cv2.cvtColor(reconstructed_img,cv2.COLOR_BGR2RGB)

plot_images([apple,orange,reconstructed_img],1,3,["Apple","Orange","Reconstructed"])
```

![[Pasted image 20250118223816.png]]

---

## Contours: An Outline in Computer Vision

Contours are outlines or boundaries of shapes found in an image. They are useful for shape analysis, object detection, and object recognition.  They are particularly *useful for finding objects in images where the shape is important but the color or texture is not.*

**Key Points:**

* **Definition:** A curve joining all the continuous points (along the boundary), having same color or intensity.
* **Representation:**  Contours are represented as a list of (x, y) coordinates in OpenCV.
* **Hierarchy:** Contours can be nested, representing objects within objects. This relationship is tracked in a hierarchy.
* **Not Edges:** Though related, contours are different from edges. *Edges are the sharp changes in intensity in an image, while contours are the boundaries of a shape*.  You typically *find edges first, then use them to find contours*.

**Why use Contours?**

* **Shape Analysis:**  Determine properties like area, perimeter, centroid, and aspect ratio of objects.
* **Object Detection:** Identify objects based on their shape regardless of color or texture.
* **Object Recognition:**  Use contour features as input to machine learning algorithms.

**Example (Python with OpenCV):**

```python
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

contours, hierarchy=cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of Contours: {len(contours)}")

img_with_countours=cv2.drawContours(img.copy(),contours,-1,(0,255,255),3) 

plot_images([img,img_with_countours],1,2)
```

![[Pasted image 20250121093914.png]]


The `cv2.findContours()` function in OpenCV is used to detect contours in a binary image. It takes three main arguments:

1. **`image`**: The input image (8-bit single-channel).  It should be a binary image, typically obtained after thresholding or edge detection.

2. **`mode`**:  Specifies the contour retrieval mode. It determines how the contours are organized hierarchically. The most commonly used modes are:

    * **`cv2.RETR_EXTERNAL`**: Retrieves only the outermost contours. This is useful when you are interested in the overall shapes of objects in the image, ignoring any internal details or holes.

    * **`cv2.RETR_LIST`**: Retrieves all of the contours without establishing any hierarchical relationships. Each contour is treated independently.

    * **`cv2.RETR_CCOMP`**: Retrieves all of the contours and organizes them into a two-level hierarchy. At the top level are the outer contours, and at the second level are the contours of holes within those outer contours. If there are further holes inside the second-level contours, they are placed at the top level again.

    * **`cv2.RETR_TREE`**: Retrieves all of the contours and reconstructs a full hierarchy of nested contours. This mode provides the most detailed representation of the contour relationships.

3. **`method`**: Specifies the contour approximation method. It controls how the contour points are stored. The most common methods are:

    * **`cv2.CHAIN_APPROX_NONE`**: Stores all the contour points. This provides the most accurate representation of the contour, but it can be memory-intensive.

    * **`cv2.CHAIN_APPROX_SIMPLE`**: Compresses horizontal, vertical, and diagonal segments and stores only their end points.  For example, a straight line will be stored as only two points. This significantly reduces the number of points stored, saving memory and improving processing speed.

    * **`cv2.CHAIN_APPROX_TC89_L1`**: Applies the Teh-Chin chain approximation algorithm using the L1 norm.

    * **`cv2.CHAIN_APPROX_TC89_KCOS`**: Applies the Teh-Chin chain approximation algorithm using the KCOS norm.


**Return Values:**

In OpenCV versions 4.x and later, `cv2.findContours()` returns two values:

* **`contours`**: A list of contours, where each contour is represented as a NumPy array of (x, y) coordinates of boundary points.

* **`hierarchy`**:  A NumPy array representing the contour hierarchy. It has as many rows as the number of contours. Each row contains four elements: `[Next, Previous, First_Child, Parent]`. These elements represent the indices of the next and previous contours at the same hierarchical level, the first child contour, and the parent contour, respectively. A value of -1 indicates the absence of the corresponding contour. 

### Movement-Detection using Contours

```python
import numpy as np
import cv2

cap = cv2.VideoCapture("video.av")

_, frame1 = cap.read()
_, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to create a mask of moving pixels
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the mask to fill in small holes and connect fragmented regions
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding rectangle of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filter out small contours (noise) based on area
        if cv2.contourArea(contour) < 900:
            continue

        # Draw a rectangle around the detected moving object
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read() 
    cv2.imshow('Motion Detection', frame1) #Display each frame

    # Exit if 'q' is pressed or video ends.
    if cv2.waitKey(40) == ord('q') or not ret: #Break the loop if end of file is reached
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

```
![[Pasted image 20250121101204.png]]

---

## Histograms in Computer Vision:

* **Representation:** A graphical representation of the distribution of pixel intensities in an image.  The x-axis represents the intensity values (e.g., 0-255 for an 8-bit grayscale image), and the y-axis represents the frequency of each intensity value.
* **Types:** Grayscale histograms (for intensity), color histograms (for color channels), and multi-dimensional histograms.
* **Uses:**
    * **Image Enhancement:**  Adjusting contrast and brightness.  Example: Histogram equalization distributes pixel intensities more evenly across the range, improving contrast.
    * **Image Segmentation:** Thresholding based on intensity levels to separate objects from the background.
    * **Object Recognition:** Comparing histograms of different images to determine similarity.  Color histograms can be used to identify objects regardless of their location or orientation in the image.
    * **Image Retrieval:** Searching for similar images in a database based on histogram comparison.

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("img.jpg",0) # grayscale

# using simple matplotlib:
# plt.hist(arr,bins,[min,max])
plt.hist(img.ravel(),256,(0,255))

# using OpenCV
cv2.calcHist(images,channels,mask,histSize,ranges)
hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
```

---

## Template Matching in OpenCV

Template matching in OpenCV is a technique for finding occurrences of a template image within a larger source image.  It works by sliding the template across the source image and calculating a similarity measure at each position.  The location with the highest similarity is considered the best match.

**Key aspects:**

* **Methods:**  OpenCV provides several matching methods (e.g., `cv2.TM_CCOEFF_NORMED`, `cv2.TM_SQDIFF_NORMED`) which offer different ways to calculate the similarity score.
* **Result:** The output is a grayscale image where brighter regions indicate higher similarity.  The location of the brightest pixel corresponds to the top-left corner of the best match.
* **Limitations:** Sensitive to scale and rotation changes.  The template and the object in the source image must have similar size and orientation for effective matching.
* **Use Cases:** Object detection in simple scenarios where the object's appearance is relatively consistent.


```python
import cv2
import numpy as np

# Load source and template images
img = cv2.imread("source_image.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template_image.jpg", cv2.IMREAD_GRAYSCALE)

# Perform template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find the location of the best match
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the matched region
h, w = template.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, 255, 2)

# Display the result
cv2.imshow("Matched Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

*Different methods for template matching*

![[Pasted image 20250121193159.png]]

![[Pasted image 20250121193903.png]]
