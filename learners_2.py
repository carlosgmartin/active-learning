import numpy as np



class Learner:



	def __init__(self, points, labels):
		self.points = points
		self.labels = labels



	def learn(self, point, label):
		return self



	def probability(self, point, label):
		''' Probability that a point has a label '''
		return 1 / len(self.labels)



	def predict(self, point):
		''' Predicted label of a point '''
		return max(self.labels, key=lambda label: self.probability(point, label))



	def expected_correct(self):
		''' Expected number of correct labels '''
		return sum(self.probability(point, self.predict(point)) for point in self.points)



	def query(self):

		# Query a random point
		# return self.points[np.random.choice(self.points.shape[0])]

		# Query the point that maximizes the expected number of correct labels
		return max(
			self.points,
			# Expected number of correct labels given the label of this point
			key=lambda point: np.sum(
				# Probability that this point has this label
				self.probability(point, label)
				# Expected number of correct labels given that this point has this label
				* self.learn(point, label).expected_correct()
				for label in self.labels
			)
		)





























