import cv2
from matplotlib import pyplot as plt
from preprocessing_image import image_gray, image_bin_adaptive, image_otsu_treshold, remove_noise
PATH = './images/test.png'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():
    img_core = load_image(PATH)
    img = image_bin_adaptive(image_gray(img_core))
    img = image_otsu_treshold(img)
    img = remove_noise(img)
    display_image(img)

if __name__ == "__main__":
    main()
