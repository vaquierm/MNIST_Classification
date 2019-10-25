# This file contains all logic to extract the largest number from a given image
import cv2
import numpy as np

from src.config import MNIST_PIXEL
from src.util.fileio import show_images, show_image


class Rectangle:

    def __init__(self, rect):

        self.box = cv2.boxPoints(rect).astype("float32")

        self.h = int(rect[1][1])
        self.w = int(rect[1][0])

        # If the angle is negative, we need to reorder the points and interchange the width and height
        if self.h < self.w and rect[2] < 0:
            # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
            temp_point = self.box[0].copy()
            self.box[0] = self.box[1]
            self.box[1] = self.box[2]
            self.box[2] = self.box[3]
            self.box[3] = temp_point

            temp_point = self.h
            self.h = self.w
            self.w = temp_point

        self.area = self.w * self.h


def __argmax(l: list, key):
    """
    Get the max arg of the list based on the lambda
    :param l: list
    :param key: lambda expression 'a -> number
    :return: The max arg based on the key
    """
    max = float('-inf')
    max_i = -1
    for i in range(len(l)):
        if key(l[i]) > max:
            max = key(l[i])
            max_i = i
    return max_i


def __extract_rectangle(img: np.ndarray, rectangle: Rectangle):
    """
    Extract the rectangle out of the np image
    :param img: The source image containing all numbers
    :param rectangle: the rectangle to extract
    :return: np array representation of the rectangle to extract
    """
    dst_pts = np.array([[0, rectangle.h - 1],
                        [0, 0],
                        [rectangle.w - 1, 0],
                        [rectangle.w - 1, rectangle.h - 1]], dtype="float32")

    src_pts = rectangle.box

    # calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (rectangle.w, rectangle.h))

    #if warped.shape[0] * 2 < warped.shape[1]: #TODO Sometimes the image is sideways happens avout 0.6% of the time
        #show_image(warped)

    canvas = np.zeros((MNIST_PIXEL, MNIST_PIXEL), dtype=np.uint8)
    filled_max_side = MNIST_PIXEL - 4
    if warped.shape[0] >= warped.shape[1]:
        h_w_ratio = warped.shape[0] / warped.shape[1]

        warped = cv2.resize(warped, (max(int(filled_max_side / h_w_ratio), 1), filled_max_side))

        x_start = int(MNIST_PIXEL / 2 - warped.shape[1] / 2)
        x_end = x_start + warped.shape[1]
        canvas[2:MNIST_PIXEL-2, x_start:x_end] = warped
    else:
        h_w_ratio = warped.shape[0] / warped.shape[1]
        warped = cv2.resize(warped, (filled_max_side, max(int(filled_max_side * h_w_ratio), 1)))

        y_start = int(MNIST_PIXEL / 2 - warped.shape[0] / 2)
        y_end = y_start + warped.shape[0]
        canvas[y_start:y_end, 2:MNIST_PIXEL - 2] = warped

    return canvas


def extract_k_numbers(img: np.ndarray, k: int = 3, show_imgs: bool = False):
    """
    Extracts k numbers present in the image
    :param img: Image to extract numbers from
    :param k: Number of numbers to be extracted from the image
    :param show_imgs: If true, debug images will be shown to the user
    :return: (k, 28, 28) np array of k numbers extracted from the picture
    """

    images_to_show = [img]
    images_titles = ["Original image"]

    # Get the to_zero thresholded image used to crop the numbers out
    img_TOZERO = cv2.threshold(img, 220, 255, cv2.THRESH_TOZERO)[1]

    # First threshold the image such that the numbers are all 255 and rest is 0
    img_BINARY = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1]

    images_to_show.append(img_BINARY)
    images_titles.append("Thresholded image")

    # Dilate slightly for disconnected components to get together
    img_DIALATE = cv2.dilate(img_BINARY, np.ones((3, 3), np.uint8), iterations=1)

    images_to_show.append(img_DIALATE)
    images_titles.append("Dilated image")

    # Get all the contours (https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html)
    # We only want extreme contours and the contours encoded as a few points
    _, contours, _ = cv2.findContours(img_DIALATE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are not enough contours, we need to erode the picture until we can get at least k contours
    while len(contours) < k:
        _, contours, _ = cv2.findContours(img_BINARY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_BINARY = cv2.erode(img_BINARY, np.ones((3, 3), np.uint8), iterations=1)


    img_color = cv2.cvtColor(img_TOZERO, cv2.COLOR_GRAY2BGR)
    rectangles = []
    # For each contour we want to calculate a metric for its size
    # to pick which rectangle we want to return
    for contour in contours:

        # Get the minimum area possibly rotated rectangle enclosing the chain point set
        rectangles.append(Rectangle(cv2.minAreaRect(contour)))

    extracted_images = np.empty((k, MNIST_PIXEL, MNIST_PIXEL))

    # Get the k biggest rectangles
    for i in range(k):
        rect_i = __argmax(rectangles, lambda r: r.area)

        # Get the biggest rectangle
        largest_rect = rectangles[rect_i]
        
        # Remove the biggest rectangle from the list
        rectangles.remove(largest_rect)

        cv2.drawContours(img_color, [np.int0(largest_rect.box)], 0, (0, 0, 255), 1)

        extracted_images[i] = __extract_rectangle(img_TOZERO, largest_rect)

    images_to_show.append(img_color)
    images_titles.append("Numbers extracted")

    if show_imgs:
        show_images(images_to_show, images_titles)
        images_to_show = []
        for i in range(k):
            images_to_show.append(extracted_images[i])
        if k > 1:
            show_images(images_to_show)
        else:
            show_image(images_to_show[0])

    return extracted_images
