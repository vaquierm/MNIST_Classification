# This file contains all logic to extract the largest number from a given image
import cv2
import numpy as np

from src.config import MNIST_PIXEL
from src.util.fileio import show_images, show_image


class Rectangle:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h


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
    number_img = img[rectangle.y:rectangle.y+rectangle.h, rectangle.x:rectangle.x+rectangle.w]

    canvas = np.zeros((MNIST_PIXEL, MNIST_PIXEL), dtype=np.uint8)
    filled_max_side = MNIST_PIXEL - 4
    if number_img.shape[0] >= number_img.shape[1]:
        h_w_ratio = number_img.shape[0] / number_img.shape[1]

        if (max(int(filled_max_side / h_w_ratio), 1), filled_max_side) == (number_img.shape[1], number_img.shape[0]):
            number_img = cv2.resize(number_img, (number_img.shape[1] + 1, number_img.shape[0]))
        else:
            number_img = cv2.resize(number_img, (max(int(filled_max_side / h_w_ratio), 1), filled_max_side))

        x_start = int(MNIST_PIXEL / 2 - number_img.shape[1] / 2)
        x_end = x_start + number_img.shape[1]
        canvas[2:MNIST_PIXEL - 2, x_start:x_end] = number_img
    else:
        h_w_ratio = number_img.shape[0] / number_img.shape[1]

        if (filled_max_side, max(int(filled_max_side * h_w_ratio), 1)) == (number_img.shape[1], number_img.shape[0]):
            number_img = cv2.resize(number_img, (number_img.shape[1], number_img.shape[0] + 1))
        else:
            number_img = cv2.resize(number_img, (filled_max_side, max(int(filled_max_side * h_w_ratio), 1)))

        y_start = int(MNIST_PIXEL / 2 - number_img.shape[0] / 2)
        y_end = y_start + number_img.shape[0]
        canvas[y_start:y_end, 2:MNIST_PIXEL - 2] = number_img

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
    img_TOZERO = cv2.threshold(img, 225, 255, cv2.THRESH_TOZERO)[1]

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

        # Get the rectangles bounding the chain set
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append(Rectangle(x, y, w, h))

    extracted_images = np.empty((k, MNIST_PIXEL, MNIST_PIXEL))

    # Get the k biggest rectangles
    for i in range(k):
        rect_i = __argmax(rectangles, lambda r: r.area)

        # Get the biggest rectangle
        largest_rect = rectangles[rect_i]
        
        # Remove the biggest rectangle from the list
        rectangles.remove(largest_rect)

        cv2.rectangle(img_color, (largest_rect.x,largest_rect.y), (largest_rect.x+largest_rect.w,largest_rect.y+largest_rect.h), (255, 0, 0), 1)

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


def extract_3_and_paste(img: np.ndarray):
    """
    Extract 3 numbers from the input image, create a long image will all 3 extracted image pasted
    next to each other.
    :param img: Input image with all numbers
    :return: (6, 28, 3 * 28)
    """
    extracted = extract_k_numbers(img, k=3)

    all_perms = np.empty((6, MNIST_PIXEL, 3 * MNIST_PIXEL))
    all_perms[0] = np.hstack((extracted[0], extracted[1], extracted[2]))
    all_perms[1] = np.hstack((extracted[0], extracted[2], extracted[1]))
    all_perms[2] = np.hstack((extracted[1], extracted[2], extracted[0]))
    all_perms[3] = np.hstack((extracted[1], extracted[0], extracted[2]))
    all_perms[4] = np.hstack((extracted[2], extracted[0], extracted[1]))
    all_perms[5] = np.hstack((extracted[2], extracted[1], extracted[0]))
    
    return all_perms
