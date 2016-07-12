import cv2
import numpy as np


class TooHigh(Exception):

	def __init__(self):
		pass

class EigenFaceModel(object):
	"""
	Class for Eigen Face model
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

	def get_cv2_EigenFaceRecognizer(self):
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


	def project_face(self, face):
		"""
		Projects face onto eigenspace

		face: face image to be projected

		returns: projected vector (face)
		"""
		mean = self.get_mean()
		#print face.shape
		face = reshape_face(face)
		vectors = self.get_eigenvectors()
		c = []
		for v in vectors:
			print "Face sha[e " + str(face.shape)
			print "mean shape " + str(mean.shape)
			c.append(v.dot(face-mean))
		return np.array(c)

	def get_scores(self, face, threshold, training=None):
		"""
		Get all scores for given face

		face: width*height-dimensional vector which represents face
		training: training set of images (reshaped)

		returns: list of scores for each image in training set. NOTE: list is ordered like images in training set
		"""
		distances = []
		f = self.project_face(face)
		if training == None:
			training = self.images
		for img in training:
			t = self.project_face(img)
			distances.append(np.sqrt(np.sum((f-t)**2)))
		scores = []
		for d in distances:
			if d >= threshold:
				d = threshold - 1
			scores.append(np.log(threshold - d) - np.log(threshold))
		return scores

	def score(self, face, ttt, threshold):
		"""
		Scores face from training set

		face: reshaped vector which represents face
		ttt: face to compare with (as images/matrices)

		returns: score (logarithm of probability)

		"""
		f = self.project_face(face)
		t = self.project_face(ttt)
		print "Calculating distance...."
		d = np.sqrt(np.sum((f-t)**2))
		print "Distance: " + str(d)
		if d >= threshold:
			d = threshold - 1
		return np.log(threshold - d) - np.log(threshold)

	def score_projected(self, f, t, threshold):
		"""
		Scores projected face from training set of projected faces

		f: projected face (using project_face(...) function)
		t: face to compare with (projected)

		returns: score (logarithm of probability)

		"""
		d = np.sqrt(np.sum((f-t)**2))
		if d >= threshold:
                        raise TooHigh
			d = threshold - 1
		return np.log(threshold - d) - np.log(threshold)


def load_face(path):
	"""
	Loads face as matrix

	returns: Matrix which represents grayscale loaded images (face)
	"""
	return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def reshape_face(face):
	"""
	Reshapes face as array

	returns: reshaped numpy.array (r*c)
	"""
	if len(face.shape) == 1:
		return face
	r,c = face.shape
	return face.reshape(r*c)
