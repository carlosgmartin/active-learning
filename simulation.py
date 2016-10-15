from __future__ import division
from learners import Learner
from sklearn.datasets import load_digits
import numpy as np
from random import shuffle
from itertools import count

# Load the dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Create the learner
learner = Learner(dataset.data, np.unique(dataset.target))


# Steps to skip before testing accuracy
skip = 5

for step in count():
    # Let the learner choose a point to label
    point = learner.choose()

    # Acquire the label of this point
    label = dataset.target[point]

    # Learn the label of this point
    learner.learn(point, label)

    if step % skip == 0:
        print 'Queried {}/{} labels so far ({:.2f}%)'.format(step, len(dataset.target), step/len(dataset.target)*100)

        # Print the ratio of correct predictions
        accuracy = np.sum(learner.predict() == dataset.target) / len(dataset.target)
        
        print 'Accuracy: {:.2f}%\n'.format(accuracy * 100)








