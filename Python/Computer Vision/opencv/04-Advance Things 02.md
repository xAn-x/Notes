## Hough Transform 

The Hough Transform is a popular technique to detect any shape, if u can represent that shape in a mathematical form.

It can detect the shape even if it is broken or distorted a little bit.

**How it works (for lines):**

1. **Edge Detection:** The Hough Transform typically operates on an edge-detected image (e.g., using Canny edge detection).
    
2. **Parameter Space:** Instead of representing lines in the image space (x, y), the Hough Transform represents them in a parameter space. For lines, a common parameterization is the polar coordinate system , where  is the perpendicular distance from the origin to the line and  is the angle between this perpendicular line and the x-axis.
    
3. **Voting:** Each edge point in the image space "votes" for all possible lines that could pass through it in the parameter space. This voting is done by incrementing the values in an accumulator array that represents the parameter space.
    
4. **Peak Detection:** After all edge points have voted, peaks in the accumulator array correspond to lines in the image space. The higher the peak, the more evidence there is for the corresponding line.

---

### How Hough transforms work

**1. Representing Lines:**

A line in the Cartesian coordinate system (x, y) can be represented by the equation:

$y = mx + c$

where $m$ is the slope and $c$ is the y-intercept.  However, this representation has a problem: vertical lines (where $m$ is infinite) cannot be represented.

The Hough Transform uses a more robust representation: the polar coordinate system $(r, \theta)$.  Here, a line is represented by:

$r = x\cos(\theta) + y\sin(\theta)$

where $r$ is the perpendicular distance from the origin to the line, and $\theta$ is the angle between this perpendicular line and the x-axis. This representation can handle all lines, including vertical ones.

**2. The Accumulator Array:**

The accumulator array is a 2D array (or histogram) that represents the parameter space $(r, \theta)$. The dimensions of the array are determined by the desired resolution. For example:

* **$r$ axis:** Discretized from $-r_{max}$ to $+r_{max}$, where $r_{max}$ is the maximum possible distance from the origin to a line in the image.
* **$\theta$ axis:** Discretized from $0$ to $\pi$ radians (or $0$ to $180$ degrees).

Each cell in the accumulator corresponds to a specific $(r, \theta)$ pair, representing a particular line.

**3. The Voting Process:**

For each edge point $(x, y)$ detected in the image:

* Iterate through all possible values of $\theta$ (from $0$ to $\pi$).
* For each $\theta$, calculate the corresponding $r$ using the equation $r = x\cos(\theta) + y\sin(\theta)$.
* Increment the cell in the accumulator array corresponding to the calculated $(r, \theta)$ pair.

This process is essentially "voting."  Each edge point votes for all possible lines that could pass through it.

**4. Peak Detection:**

After all edge points have voted, local maxima (peaks) in the accumulator array represent lines in the image space.  The higher the peak, the more edge points lie on the corresponding line, indicating stronger evidence for its presence.

**5.  From Parameter Space back to Image Space:**

Once the peaks are identified in the accumulator array, the corresponding $(r, \theta)$ values are used to determine the equation of the line in the image space (x, y).  This can be done using the polar equation of the line or by converting it back to Cartesian coordinates.

**Why does this work?**

Points that lie on the same line in the image space will contribute to the same $(r, \theta)$ cell in the parameter space.  Therefore, collinear points will generate a peak in the accumulator array.

**Advantages of the Hough Transform:**

* Robust to noise and gaps in lines.
* Can detect multiple lines simultaneously.
* Relatively insensitive to occlusion.


**Limitations:**

* Computationally expensive, especially for higher resolutions and more complex shapes.
* Parameter tuning can be challenging. The choice of resolution for the accumulator array affects the accuracy and performance.

 
```python
import cv2
import numpy as np

# Load image and perform edge detection
img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough Transform
#cv2.HoughLines(image,rho,theta,threshold)
# rho: dist resolution of accum in pixel
# tehra: angle reso. of accum in radians
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)  # Adjust parameters as needed

# Draw detected lines
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho 
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow("Lines Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![[Pasted image 20250121195853.png]]
### Probabilistic Hough Transform

The Probabilistic Hough Transform is an optimization of the standard Hough Transform, designed to be more efficient and less computationally intensive.  It doesn't consider every single edge point for voting, but rather a random subset, making it significantly faster.

**Key Differences from Standard Hough Transform:**

* **Random Subsampling:** Instead of iterating through all edge points, the Probabilistic Hough Transform selects a random subset of edge points for voting in the accumulator array.  This significantly reduces the computational burden.

* **Line Segment Detection:** The Probabilistic Hough Transform directly detects line segments, rather than infinite lines.  It achieves this by stopping the voting process for a line once a sufficient number of votes have been accumulated or a predefined minimum line length is reached.

* **Parameters:**  In addition to `rho` and `theta`, the Probabilistic Hough Transform introduces two new parameters:
    * **`minLineLength`:** The minimum length of a line segment to be considered.  Shorter segments are discarded.
    * **`maxLineGap`:** The maximum allowed gap between points on the same line to link them together.  Larger gaps will result in separate line segments.

**Advantages over Standard Hough Transform:**

* **Faster:** Significantly less computation due to subsampling.
* **Line Segments:** Directly detects line segments, which is often more useful in practice than infinite lines.
* **Less Memory:** Requires a smaller accumulator array due to the focus on line segments rather than the entire parameter space.

**OpenCV Implementation (`cv2.HoughLinesP()`):**

```python
import cv2
import numpy as np

# Load image and perform edge detection
img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Probabilistic Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
   
# Draw detected line segments
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow("Line Segments Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

The `cv2.HoughLinesP()` function returns an array where each element is a 4-element vector $(x_1, y_1, x_2, y_2)$ representing the endpoints of a detected line segment.  The `threshold`, `minLineLength`, and `maxLineGap` parameters control the sensitivity and characteristics of the line detection.

![[Pasted image 20250121200404.png]]

### OpenCV DNN Module:

The DNN (Deep Neural Network) module in OpenCV is a powerful component that allows users to load and run pre-trained deep learning models for various tasks such as image classification, object detection, and segmentation. It provides a unified interface to work with models trained using popular deep learning frameworks like TensorFlow, Caffe, and PyTorch.

> [!Note]
> DNN module only supports deep learning inference on images and videos. It does not support fine-tuning and training

#### Key Features

- Can import model from wide no. of frameworks may it be pytorch, tensorflow, onnx etc.
- Fast Inference in Intel CPU or GPU as DNN is optimised for perfomance & can leverage them hardware.
- Vast no. of methods for Pre/Post processing data.
- Cross-platform support.

```python
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

net=cv2.dnn.readNet(model="weights-file",config="config file for Tf and Caffe",frameWork="Caffe|TensorFlow|PyTorch..")

# or u can import from a specific frameWork
net=cv2.dnn.readNetFroTorch(".pth file")

# DNN don't take image input directly, we need to preprocess it 
img=cv2.imread("image")
blob=cv2.dnn.blobFromImage(image=image,scalefactor=0.01,size=(224,224),mean=(),std=(),swapRB=True) # -> shape:[1,3,224,224], auto batched

model.setInput(blob)
outputs=model.forward() # forward-pass

```