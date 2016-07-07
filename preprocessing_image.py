import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin

def image_otsu_treshold(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def dilate(image, value):
    kernel = np.ones((value,value))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image, value):
    kernel = np.ones((value,value))
    return cv2.erode(image, kernel, iterations=1)

def invert(image):
    return 255-image

def remove_noise(binary_image):
    ret_val = erode(dilate(binary_image, 2), 3)
    ret_val = invert(ret_val)
    ret_val = erode(dilate(binary_image, 3), 3)
    return ret_val
