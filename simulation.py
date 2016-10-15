from __future__ import division
from learners import Learner
from sklearn.datasets import load_digits
import numpy as np
from random import shuffle

# Load the dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Create the learner
learner = Learner(dataset.data, np.unique(dataset.target))

# Perform semi-supervised learning
sequence = list(enumerate(dataset.target))
shuffle(sequence)

# Steps to skip before testing accuracy
skip = 5

for step, (point, label) in enumerate(sequence):

    # Learn the label of this data point    
    learner.learn(point, label)

    if step % skip == 0:

        print 'Queried {}/{} labels so far ({:.2f}%)'.format(step, len(sequence), step/len(sequence)*100)

        # Print the ratio of correct predictions
        accuracy = np.sum(learner.predict() == dataset.target) / len(dataset.target)
        
        print 'Accuracy: {:.2f}%\n'.format(accuracy * 100)





