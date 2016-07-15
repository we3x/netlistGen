import cv2
from matplotlib import pyplot as plt
import numpy as np
from preprocessing_image import image_bin, image_gray, invert, dilate, select_roi
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

    eigenModel = EigenComponentModel(train_images, train_labels)

    for x,y,w,h in regions:
        component = img_core[y:y+h, x:x+w]
        component = cv2.resize(component, (128,128))
        component = cv2.cvtColor(component, cv2.COLOR_BGR2GRAY)
        s = eigenModel.get_scores(component, 100000)
        print(s)


    display_image(img_core_selected)

if __name__ == "__main__":
    main()
