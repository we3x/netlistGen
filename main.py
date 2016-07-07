PATH = './images/test.png'

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def main():
    img = load_image(PATH)
    display_image(img)

if __name__ == "__main__":
    main()
