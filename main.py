import cv2
from matplotlib import pyplot as plt
import numpy as np
from preprocessing_image import image_bin, image_gray, invert, dilate, select_roi, clear_component, search_line, erode
from recognize import EigenComponentModel
PATH = './images/tests/test5.png'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():
    train_images = []
    labels = ['and', 'nand', 'nor', 'not', 'xnor', 'or', 'xor']
    train_labels = [0, 1, 2, 3, 4, 5, 6]

    for label in labels:
        img = load_image('./images/components/'+label+'.png')
        img = cv2.resize(img, (128, 128))
        train_images.append(img)

    img_core = load_image(PATH)
    img_core_bin = image_bin(image_gray(img_core))
    img_core_invert = invert(img_core_bin)
    img_core_dilate = dilate(img_core_invert, 1)
    img_core_selected, component, regions = select_roi(img_core.copy(), img_core_dilate)

    img_core_bin_clear = img_core_bin.copy()
    for x,y,w,h in regions:
        img_core_bin_clear = clear_component(img_core_bin_clear, x, y, w, h)
    dots = []
    x = img_core_bin_clear.shape[0]
    y = img_core_bin_clear.shape[1]
    img_core_bin_clear = erode(img_core_bin_clear, 3)
    plt.subplot(2, 1, 1)
    plt.imshow(img_core_bin_clear.copy(), 'gray')
    for a in range(x):
        for b in range(y):
            if(img_core_bin_clear[a][b] == 0):
                dots.append(search_line(img_core_bin_clear, a, b))

    for dot in dots:
        x1,y1 = dot[0]
        x2,y2 = dot[1]
        img_core_bin_clear[x1][y1] = 0
        img_core_bin_clear[x2][y2] = 0
        print(dot)
        cv2.line(img_core_bin_clear, (y1,x1), (y2,x2), 0)


    plt.subplot(2, 1, 2)
    plt.imshow(img_core_bin_clear, 'gray')

    plt.show()

if __name__ == "__main__":
    main()
