import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

class TooHeigh(Exception):
    def __init__(self):
        pass

class EigenComponentModel(object):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.model = cv2.face.createEigenFaceRecognizer()
        self.model.train(images, np.array(labels))

    def get_cv2_EigenComponentRecognizer(self):
        return self.model

    def get_mean(self):
        mean = self.model.getMean()
        return mean.reshape(mean.shape[1])

    def get_eigenvectors(self):
        vectors = self.model.getEigenVectors()
        return vectors.reshape(vectors.shape[0], vectors.shape[1]).T

    def project_component(self, component):
        mean = self.get_mean()
        component = reshape_component(component)
        vectors = self.get_eigenvectors()
        c = []
        for vector in vectors:
            c.append(vector.dot(component - mean))
        return np.array(c)

    def get_scores(self, component, treshold, training=None):
        distances = []
        f = self.project_component(component)

        if training == None:
            training = self.images
        for img in training:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = self.project_component(img)
            distances.append(np.sqrt(np.sum((f-t)**2)))
        scores = []
        for distance in distances:
            if distance >= treshold:
                distance = treshold -1
            scores.append(np.log(treshold - distance) - np.log(treshold))
        return scores

    def score(self, component, ttt, treshold):
        f = self.project_component(component)
        t = self.project_component(ttt)
        d = np.sqrt(np.sum((f-t)**2))
        if d >= treshold:
            d = treshold - 1
        return np.log(treshold-d) - np.log(treshold)

    def score_projected(self, f, t, treshold):
        d = np.sqrt(np.sum((f-t)**2))
        if d >= treshold:
            raise TooHeigh
            d = treshold - 1
        return np.log(treshold - d) - np.log(treshold)

def reshape_component(component):
    if len(component.shape) == 1:
        return component
    r,c = component.shape
    return component.reshape(r*c)
