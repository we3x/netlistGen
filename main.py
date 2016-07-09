import cv2
from matplotlib import pyplot as plt
from preprocessing_image import image_gray, image_bin_adaptive, image_otsu_treshold, remove_noise, closing
PATH = './images/test2.jpg'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():
    img_core = load_image(PATH)
    img = image_bin_adaptive(image_gray(img_core))
    img = image_otsu_treshold(img)
    cimg = closing(img)
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.subplot(122)
    plt.imshow(cimg, 'gray')
    plt.show()

if __name__ == "__main__":
    main()
