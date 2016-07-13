import cv2
from matplotlib import pyplot as plt
import numpy as np
from preprocessing_image import image_gray, image_bin_adaptive, image_otsu_treshold, remove_noise, closing, dilate, invert, select_roi, image_bin
PATH = './images/tests/test5.png'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():
    train_images = []
    labels = ['and', 'nand', 'nor', 'not', 'xnor', 'or', 'xor']

    for label in labels:
        img = load_image('./images/components/'+label+'.png')
        train_images.append(img)

    for img in train_images:
        display_image(img)


    img_core = load_image(PATH)
    img_core_bin = image_bin(image_gray(img_core))
    img_core_invert = invert(img_core_bin)
    img_core_dilate = dilate(img_core_invert, 1)
    img_core_selected, component, regions = select_roi(img_core.copy(), img_core_dilate)
    display_image(img_core_selected)

if __name__ == "__main__":
    main()
