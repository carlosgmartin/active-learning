import numpy as np
from itertools import count
from learner import Learner
from sklearn.datasets import load_digits

# Load dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Define the set of points
points = dataset.data

# Define the set of labels
labels = np.unique(dataset.target)

# Assign a label to each point
def get_label(point):
	for other_point, label in zip(dataset.data, dataset.target):
		if (other_point == point).all():
			return label

# Create the learner
learner = Learner(points, labels)

for queries in count():

	# Query the label of a point
	point = learner.query()

	# Get the label of this point
	label = get_label(point)

	# Learn the label of this point
	learner = learner.learn(point, label)

	# Get the predicted labels of all points
	predictions = map(learner.predict, points)

	# Compute the number of correct predictions
	correct = sum(predictions == dataset.target)

	print 'Queries:'.ljust(20) + str(queries)
	print 'Correct:'.ljust(20) + str(correct)
	print ''
















