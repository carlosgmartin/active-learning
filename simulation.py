# python -m cProfile -s time simulation.py


from __future__ import division
from learners import Learner
from sklearn.datasets import load_digits
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


# Load the dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Steps to skip before testing accuracy
skip = 20



for trial in range(1):

    print 'Trial {}\n'.format(trial)

    # Create the learner
    learner = Learner(dataset.data, np.unique(dataset.target))

    steps = []
    accuracies = []

    for step in range(len(dataset.target)):
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

            steps.append(step)
            accuracies.append(accuracy)

    plt.plot(steps, accuracies, linewidth=2)





plt.xlabel('Queries')
plt.ylabel('Accuracy')
plt.show()





