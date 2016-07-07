import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
