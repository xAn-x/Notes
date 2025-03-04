# utils.py

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_images(images: list[np.ndarray], rows: int, cols: int, titles: list[str] = None, figsize: tuple[int, int] = (10, 10)) -> None:
    """
    Plot multiple images in a grid layout.

    Parameters
    ----------
    images : list[numpy.ndarray]
        List of images to be displayed.
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    titles : list[str], optional
        List of titles to be used for each image.
    figsize : tuple[int, int], optional
        Size of the figure in inches.

    Returns
    -------
    None
    """
    if titles is not None:
        assert len(images) == len(titles), "Number of titles must be equal to number of images"

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 2 or img.shape[2] == 1:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            if titles is not None:
                ax.set_title(titles[i])
            ax.axis('off')  # Hide axis
        else:
            ax.set_axis_off()  # Hide unused subplots

    plt.show()



def get_pixel_info(img: np.ndarray, window_name: str = "Image") -> list[tuple[int, int,np.ndarray]]:
    """
    Display an image in a window and capture pixel coordinates on mouse click.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be displayed where pixel values can be selected.
    window_name : str, optional
        The name of the window in which the image will be displayed (default is "Image").

    Returns
    -------
    list of tuple of int
        A list of tuples containing the (x, y,color[])
    """

    img=img.copy()
    points = []
    window=cv2.namedWindow(window_name)
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y, img[x,y,:]))
            cv2.rectangle(img,(x,min(y+12,img.shape[0])),(min(img.shape[1],x+130),max(0,y-12)),(0,0,0),-1)
            cv2.putText(img, f"({x}, {y}, {points[-1][-1]})", (x, y), cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)

            cv2.imshow(window_name, img)

    while True:
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, mouse_callback)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return points

class PYRAMID_DIRN(Enum):
    Down=0
    Up=1

def generate_k_pyramid(img: np.ndarray, k: int,dirn:PYRAMID_DIRN)->list[np.ndarray]:
    """
    Generate a pyramid of images by applying either pyrDown or pyrUp on the input image.

    Parameters
    ----------
    img : numpy.ndarray
        The input image
    k : int
        The number of pyramid levels to generate
    dirn : PYRAMID_DIRN
        The direction of the pyramid generation, either Down to reduce the image size or Up to increase the image size

    Returns
    -------
    list of numpy.ndarray
        A list of images representing the pyramid, the first element is the input image itself, and the subsequent elements are the result of applying the pyramid generation method to the previous element
    """
    img=img.copy()
    method=cv2.pyrDown if dirn==PYRAMID_DIRN.Down else cv2.pyrUp

    pyramid=[img]
    for i in range(k):
        img=method(img)
        pyramid.append(img)

    return pyramid


def generte_laplacianPyr_from_gaussianPyr(gaussianPyr: list[np.ndarray]) -> list[np.ndarray]:
    """
    Generate the Laplacian Pyramid from the Gaussian Pyramid.

    Parameters
    ----------
    gaussianPyr : list[numpy.ndarray]
        A list of images representing the Gaussian Pyramid.

    Returns
    -------
    list[numpy.ndarray]
        A list of images representing the Laplacian Pyramid.

    Raises
    ------
    ValueError
        If the number of images in the Gaussian Pyramid is not greater than 1.
    """
    n=len(gaussianPyr)
    if n <= 1:
        raise ValueError("Gaussian Pyramid must have at least 2 images")

    laplacianPyr = [gaussianPyr[-1]]

    for i in range(n - 1, 0, -1):
        gaus_expanded = cv2.pyrUp(gaussianPyr[i])
        laplacianPyr.append(cv2.subtract(gaussianPyr[i - 1], gaus_expanded))

    return laplacianPyr
