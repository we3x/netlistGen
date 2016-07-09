import cv2
from matplotlib import pyplot as plt
import numpy as np
from preprocessing_image import image_gray, image_bin_adaptive, image_otsu_treshold, remove_noise, closing, dilate, invert, select_roi, image_bin
PATH = './images/tests/test10.png'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():

    img_core = load_image(PATH)
    train_bin = image_bin(image_gray(img_core))
    train_bin = invert(train_bin)
    train_bin = dilate(train_bin, 1)
    display_image(train_bin)
    selected_regions, component = select_roi(img_core.copy(), train_bin)
    display_image(selected_regions)

if __name__ == "__main__":
    main()
