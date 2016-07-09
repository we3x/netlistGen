import cv2
import numpy as np
from matplotlib import pyplot as plt
import collections

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin

def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin

def image_otsu_treshold(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def dilate(image, value):
    kernel = np.ones((value,value))
    return cv2.dilate(image, kernel, iterations=3)

def erode(image, value):
    kernel = np.ones((value,value))
    return cv2.erode(image, kernel, iterations=1)

def invert(image):
    return 255-image

def remove_noise(binary_image):
    ret_val = erode(dilate(binary_image, 3), 3)
    return ret_val

def closing(binary_image):
    ret_val = dilate(erode(binary_image, 3), 3)
    return ret_val

def select_roi(image_orig, image_bin):

    im2, contours, hierarchy = cv2.findContours(image_bin.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_regions = []
    regions_dic = {}
    region_borders = []
    i = -1

    plt.imshow(im2)
    plt.show()

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        print('here')
        borders = [x,y,w,h]
        region_borders.append(borders)
        region = image_bin[y:y+h+1,x:x+w+1]
        regions_dic[x] = region
        cv2.rectangle(image_orig,(x-3,y-3),(x+w+3,y+h+3),(255,0,0),1)

    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()

    return image_orig, sorted_regions
