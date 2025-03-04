Open-CV is a free open-source lib used in computer-vision, helps in image processing, object-detection, key-points detections etc. 

```sh
pip install opencv-python
```

Open-CV uses numpy array to read images. It provides many helper method for converting, editing and manipulating images.

## Working with Images

```python
import cv2

# Reading-image
# shape:(H,W,C), BGR (default)
img=cv2.imread('image-name',cv2.IMG_COLOR)

print(type(img)) # nd-array

# Displaying Image
cv2.imshow(img)
cv2.waitkey(5000) # wait for 5 sec before closing window
# use 0 to wait until closed
cv2.destroyAllWindows()


# Write images
cv2.imwrite("image-name.jpg",img)


# shape:
h,w,c=img.shape
```

## Working With Videos

```python
import cv2

# Read a video-file
cap=cv2.VideoCapture("video.mp4")
# Read Video from camera
cap=cv2.VideoCapture(0) # use 0th device to read

# Write to a video-file

# specify 4-char-code for compression and decompression
fourcc=cv2.VideoWriter_fourcc(*'XVID')
# specify where to save
frame_rate=20,
dimn=(640,480)
out=cv2.VideoWriter("out.avi",fourcc,frame_rate,dimn)


while cap.isOpenend():
	# cap.get(cv2.CAP_PROP_{{property_name}})
	fps=cap.get(cv.CAP_PROP_FPS)
	total_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)	
	
	ret,frame=cap.read()
	# if it return a frame
	if ret:
		# use get() to get different values
		width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print(width,height)
		
		# use set() to set different values
		cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
	
		# convert BGR-to-GreyScale
		grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		out.write(grayscale)
		
		# if 'q'->exit
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break	
	else:
		break

# release resources
out.release()
cap.release()
cv2.destroyAllWindows()

```

## Drawing Geometric Shapes in OpenCV

OpenCV provides functions to draw various geometric shapes directly onto images.  This is useful for annotations, visualizations, and image manipulation.

All the edits are in-place.

| Shape     | Function                                                                        | Description                                                                              |
| --------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Line      | `cv2.line(img, pt1, pt2, color, thickness)`                                     | Draws a line segment connecting `pt1` and `pt2`.                                         |
| Rectangle | `cv2.rectangle(img, pt1, pt2, color, thickness)`                                | Draws a rectangle with top-left corner at `pt1` and bottom-right corner at `pt2`.        |
| Circle    | `cv2.circle(img, center, radius, color, thickness)`                             | Draws a circle with given center and radius.                                             |
| Ellipse   | `cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)` | Draws an ellipse.  `axes` specifies half the size of the ellipse's major and minor axes. |
| Polygon   | `cv2.polylines(img, pts, isClosed, color, thickness)`                           | Draws a polygon defined by an array of points `pts`.                                     |
| Text      | `cv2.putText(img, text, org, font, fontScale, color, thickness)`                | Renders text on the image. `org` is the bottom-left corner of the text string.           |

**Code Example:**

```python
import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Draw a green line
cv2.line(img, (0, 0), (511, 511), (0, 255, 0), 5)  # Green, thickness 5

# Draw a blue rectangle
cv2.rectangle(img, (384, 0), (510, 128), (255, 0, 0), 3)  # Blue, thickness 3

# Draw a red circle
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)  # Red, filled circle

# Add text
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,"kaBoom",font,4,(0,0,255),2)

# Display the image
cv2.imshow("Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
## Events in Open-CV

Events in OpenCV refer to actions like mouse clicks, keyboard presses, and trackbar movements that occur within a displayed OpenCV window.

```python
import cv2
import numpy as np

# list down all events
events=[i for i in dir(cv2) if "EVENT" in cv2]

# Mouse Event Eg
def mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(f"Left click at ({x}, {y})")
	elif event==cv2.EVENT_RBUTTINDOWN:
		print(f"Right click at ({x}, {y})")

img = cv2.imread('image.jpg') 
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Keyboard Event Eg:
img = cv2.imread('image.jpg') 

while True:
	cv2.imshow('Image', img)
	key = cv2.waitKey(0)
	
	if key == ord('q'): # Press 'q' to exit
		break
	elif key == ord('s'): # Press 's' to save
		cv2.imwrite('saved_image.jpg', img)
	
	cv2.destroyAllWindows()


# TrackBar event Eg
import numpy as np

def on_trackbar(val):
	print(f"Trackbar value: {val}")

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# cv2.createTrackbat=r("trackbar-name",'window',min,max,callback)
cv2.createTrackbar('B','image',0,255,on_trackbar)
cv2.createTrackbar('G','image',0,255,on_trackbar)
cv2.createTrackbar('R','image',0,255,on_trackbar)

while True:
	cv2.imshow('image',img)
	if cv2.waitkey(1)==ord('q'):
		break
	# get val for channels
	b=cv2.getTrackbarPos("B","image")
	g=cv2.getTrackbarPos("G","image")
	r=cv2.getTrackbarPos("R","image")
	
	# set color of chanel as in trackbar 
	img[:]=[b,g,r]
	
cv2.destroyAllWindows()

```
### Some Intermediate E.g.
```python
import cv2
from pydantic import BaseClass


class Circle(BaseClass):
	h:float
	k:float
	r:float=3.0

def func(event,x,y,flags,params):
	if event==cv2.EVENT_LBUTTONDOWN:
		img,circles=params
		
		# draw circle
		circle=Circle(x,y)
		circles.append(circle)
		
		cv2.circle(img,(x,y),circle.r,(0,0,255),-1)
		
		# join each circle with the last one
		for circle in circles[:-1]:
			h1,k1=circle
			cv2.line(img,(h1,k1),(x,y),(0,255,0),2)

img=cv2.imread('image.jpg')
circles=[]
cv2.setMouseCallback('image',func,img,circles)
cv2.waitkey(0)
cv2.destroyAllWindows()
```


## Some Important Functions in CV2

```python
import cv2
import numpy as np


# Color Conversions
#-----------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Converts 2 grayscale
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converts to HSV


# Image Resizing
#---------------
resized_img = cv2.resize(img, (new_width, new_height)) 
resized_img_scaled = cv2.resize(img, None, fx=0.5, fy=0.5) # Resizes by scaling factor


# Image Cropping (Region of Interest - ROI)
#-----------------------------------------
roi = img[y_start:y_end, x_start:x_end] # Extracts a rectangular region


# Splitting and Merging Image Channels
# -------------------------------------

b, g, r = cv2.split(img)  # Splits into B,G,R channels
merged_img = cv2.merge([b, g, r]) # Merges channels back into an image



# Blurring/Smoothing
#------------------

blurred_img = cv2.blur(img, (5, 5))  # Average blur
gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 0) # Gaussian blur
median_blurred = cv2.medianBlur(img, 5) # Median blur (good for salt-and-pepper noise)

```

## Image Transforming Operations

### Affine Transforms


![[Pasted image 20250124195712.png]]


![[Pasted image 20250124195043.png]]

```
tfms=[
	ScaleX, ShearX, translateX
	ShearY,, ScaleY, translateY
]
```

U can use the `getRotationMatrix2D(center,angle,scale)` to rotate an image.

```python
import cv2
import numpy as np

# translation
translate_tfms=[[1,0,translate_x],[0,1,translate_y]]

# shear
shear_tfms=[[1,shear_x,0],[shear_y,1,0]]

# scale
scale_tfms=[[scale_x,0,0],[0,scale_y,0]]

# rotation
rotation_tfms=cv2.getRotationMatrix2D(center,angle_in_deg,scale)

# apply transforms
new_imgs=cv2.wrapAffine(img,tfms)śś/cvt/c
```

## Bitwise Operations for Masking

```python
import cv2
import numpy as np

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# U can directly use the operator but i u have some mask to apply use the built-in functions

# 1. Bitwise AND (`&`)
# --------------------
# Useful for masking: isolating a region of an image
result_and = cv2.bitwise_and(img1, img2)

# Example: Creating a mask
mask = np.zeros(img1.shape[:2], dtype="uint8") # Create a black mask
cv2.rectangle(mask, (50, 50), (200, 200), 255, -1) # Draw a white rectangle on the mask
masked_image = cv2.bitwise_and(img1, img1, mask=mask) # Apply the mask


# 2. Bitwise OR (`|`)
# -------------------
# Useful for combining images, adding elements
result_or = cv2.bitwise_or(img1, img2)


# 3. Bitwise XOR (`^`)
# --------------------
# Useful for finding differences between images, or toggling pixel values
result_xor = cv2.bitwise_xor(img1, img2)
j

# 4. Bitwise NOT (`~`)
#--------------------
# Inverts the pixel values (like a color negative)
result_not1 = cv2.bitwise_not(img1)


# Display Results (optional)
cv2.imshow('AND', result_and)
cv2.imshow('OR', result_or)
cv2.imshow('XOR', result_xor)
cv2.imshow('NOT', result_not1)
cv2.imshow('Masked Image', masked_image) # Display the masked image
cv2.waitKey(0)
cv2.destroyAllWindows()

```

**Important Considerations:**

* **Image Size:**  Both images should have same shape.
* **Data Type:** Ensure your images are loaded with a suitable data type (e.g., `uint8`). 
* **Masking:** The `mask` parameter in functions like `cv2.bitwise_and()` is very powerful.  It allows you to apply the bitwise operation only to specific regions of the image.  The mask should be a single-channel (grayscale) image of the same size as the input image.  White pixels in the mask indicate the regions where the operation should be applied, and black pixels indicate regions to ignore.