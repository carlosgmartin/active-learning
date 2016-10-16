from __future__ import division
import numpy as np
from sklearn.datasets import load_digits
from itertools import count
import matplotlib.pyplot as plt
from learners_3 import Learner

# Load dataset
dataset = load_digits()

# Turn into a binary classification problem
dataset.target = dataset.target < 5

# Define the set of possible points
points = np.arange(len(dataset.data))

# Define the set of possible labels
labels = np.unique(dataset.target)


for trial in range(20):

    print 'Trial {}'.format(trial)

    # Shuffle the dataset
    state = np.random.get_state()
    np.random.shuffle(dataset.data)
    np.random.set_state(state)
    np.random.shuffle(dataset.target)

    # Create the learner
    learner = Learner(dataset.data, labels)

    if trial % 2 == 0:
        learner.query = learner.query_divergence # (deterministic)
        plot_label = 'Active sampling'
        plot_color = 'red'
    else:
        learner.query = learner.query_random
        plot_label = 'Random sampling'
        plot_color = 'blue'

    queries_list = []
    correct_list = []

    for queries in range(len(points)):

        # Query the label of a point
        point = learner.query()

        # Get the label of this point
        label = dataset.target[point]

        # Learn the label of this point
        learner.learn(point, label)

        # Get the predicted labels of all points
        predictions = map(learner.predict, points)

        # Find the number of correct predictions
        correct = sum(predictions == dataset.target)

        # print 'Queries:'.ljust(20) + '{:20.0f}{:20.2f}%'.format(queries, queries/len(points)*100)
        # print 'Correct:'.ljust(20) + '{:20.0f}{:20.2f}%'.format(correct, correct/len(points)*100)
        # print 'Expected correct:'.ljust(20) + '{:20.0f}{:20.2f}%'.format(learner.expected_correct(), learner.expected_correct()/len(points)*100)

        queries_list.append(queries/len(points))
        correct_list.append(correct/len(points))

    plt.plot(queries_list, correct_list, linewidth=1, color=plot_color, label=plot_label, alpha=.5)

plt.xlabel('Queries')
plt.ylabel('Correct')
# plt.legend()
plt.show()


