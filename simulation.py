from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from learner import Learner
from itertools import count

# Load the dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Define the set of points
points = np.arange(len(dataset.data))

# Define the set of labels
labels = np.unique(dataset.target)

# Define the total number of queries per trial
total_queries = int(len(points) * .5)

# Turn on interactive plotting
plt.ion()

# Define the x axis
plt.xlabel('Fraction of points queried')
x = np.arange(total_queries) / len(points)

# Define the y axis
plt.ylabel('Fraction of labels correct')
y = np.empty(total_queries)

for trial in count():

	# Shuffle the dataset
	data, target = shuffle(dataset.data, dataset.target)

	# Create the learner
	learner = Learner(data, labels)

	if trial % 2 == 0:
		# Use random sampling
		learner.query = learner.query_random
		color = 'blue'
	else:
		# Use active sampling
		learner.query = learner.query_active
		color = 'red'

	for queries in range(total_queries):

		# Query the label of a point
		point = learner.query()

		# Learn the label of this point
		learner.learn(point, target[point])

		# Predict the labels of all points
		predictions = map(learner.predict, points)

		# Get the number of correct predictions
		correct = sum(predictions == target)

		y[queries] = correct / len(points)

	plt.plot(x, y, alpha=.5, color=color)
	plt.pause(.1)

plt.show(block=True)





