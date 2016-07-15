import cv2
import numpy as np
from matplotlib import pyplot as plt
import collections
from operator import itemgetter

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

def merage_contours(borders):
    region_borders = []
    for i in range(len(borders)):
        x1, y1, w1, h1 = borders[i]
        for j in range(len(borders)):
            x2, y2, w2, h2 = borders[j]
            if(x1< x2 and x1 + w1 > x2 and y1 < y2 and y1 - h1 < y2 - h2 and x1 + w1 < x2 + w2):

                w1 = x2 - x1 + w2
        region_borders.append([x1, y1, w1, h1])

    return region_borders

def remove_max(borders):
    surfices = []
    ret_borders = []
    for x,y,w,h in borders:
        surfices.append((x+w)*(y+h))

    max_sur = max(surfices)
    for x,y,w,h in borders:
        if (((x+w)*(y+h)) != max_sur):
            ret_borders.append([x,y,w,h])

    return ret_borders

def select_roi(image_orig, image_bin):

    im2, contours, hierarchy = cv2.findContours(image_bin.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_regions = []
    regions_dic = {}
    region_borders = []
    regions = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x = x - 6
        y = y - 6
        w = w + 12
        h = h + 12
        borders = (x,y,w,h)
        region_borders.append(borders)

    region_borders = remove_max(region_borders)
    region_borders = merage_contours(region_borders)
    for x,y,w,h in region_borders:
        region = image_bin[y:y+h,x:x+w]
        regions_dic[x] = region
        regions.append(region)
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,0,0),1)


    sorted_regions_dic = collections.OrderedDict(sorted(regions_dic.items()))
    sorted_regions = sorted_regions_dic.values()


    return image_orig, sorted_regions, region_borders
