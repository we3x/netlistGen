import cv2
import numpy as np


class TooHigh(Exception):

	def __init__(self):
		pass

class EigenComponentModel(object):
	"""
	Class for Eigen Component model
	"""

	def __init__(self, images, labels):
		"""
		Initialization and training

		images: training set of images
		labels: labels for training images
		"""
		self.images = images
		self.labels = labels
		self.model = cv2.createEigenFaceRecognizer()
		self.model.train(images, np.array(labels))

	def get_cv2_EigenComponentRecognizer(self):
		return self.model

	def get_mean(self):
		"""
		Returns mean image as width*height-dimensional vector

		returns: numpy array which represents image
		"""
		mean = self.model.getMat("mean")
		return mean.reshape(mean.shape[1])

	def get_eigenvectors(self):
		"""
		Returns numpy array of vectors which represent eigen vectors

		returns: eigenvectors (unit vectors)
		"""
		vectors = self.model.getMat("eigenvectors")
		return vectors.reshape(vectors.shape[0], vectors.shape[1]).transpose()


	def project_component(self, component):
		"""
		Projects component into eigenspace

		component: component image to be projected

		returns: projected vector (component)
		"""
		mean = self.get_mean()
		#print component.shape
		component = reshape_component(component)
		vectors = self.get_eigenvectors()
		c = []
		for v in vectors:
			print "Component sha[e " + str(component.shape)
			print "mean shape " + str(mean.shape)
			c.append(v.dot(component - mean))
		return np.array(c)

	def get_scores(self, component, threshold, training=None):
		"""
		Get all scores for given component

		component: width*height-dimensional vector which represents component
		training: training set of images (reshaped)

		returns: list of scores for each image in training set. NOTE: list is ordered like images in training set
		"""
		distances = []
		f = self.project_component(component)
		if training == None:
			training = self.images
		for img in training:
			t = self.project_component(img)
			distances.append(np.sqrt(np.sum((f-t)**2)))
		scores = []
		for d in distances:
			if d >= threshold:
				d = threshold - 1
			scores.append(np.log(threshold - d) - np.log(threshold))
		return scores

	def score(self, component, ttt, threshold):
		"""
		Scores component from training set

		component: reshaped vector which represents component
		ttt: component to compare with (as images/matrices)

		returns: score (logarithm of probability)

		"""
		f = self.project_component(component)
		t = self.project_component(ttt)
		print "Calculating distance...."
		d = np.sqrt(np.sum((f-t)**2))
		print "Distance: " + str(d)
		if d >= threshold:
			d = threshold - 1
		return np.log(threshold - d) - np.log(threshold)

	def score_projected(self, f, t, threshold):
		"""
		Scores projected component from training set of projected components

		f: projected component (using project_component(...) function)
		t: component to compare with (projected)

		returns: score (logarithm of probability)

		"""
		d = np.sqrt(np.sum((f-t)**2))
		if d >= threshold:
                        raise TooHigh
			d = threshold - 1
		return np.log(threshold - d) - np.log(threshold)


def load_component(path):
	"""
	Loads component as matrix

	returns: Matrix which represents grayscale loaded images (component)
	"""
	return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def reshape_component(component):
	"""
	Reshapes component as array

	returns: reshaped numpy.array (r*c)
	"""
	if len(component.shape) == 1:
		return component
	r,c = component.shape
	return component.reshape(r*c)
