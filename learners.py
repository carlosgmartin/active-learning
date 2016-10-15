from __future__ import division
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import binom, beta
from scipy.integrate import quad
from scipy.special import binom as binom_coefficient
import numpy as np


class Learner:

    def __init__(self, data, labels):
        '''Initialize the learner'''

        self.labels = labels

        clustering = AgglomerativeClustering()
        clustering.fit(data)

        # Store the parent and children of each node
        self.parent = {}
        self.children = {}

        for parent, children in enumerate(clustering.children_, clustering.n_leaves_):
            self.children[parent] = children
            for child in children:
                self.parent[child] = parent

        for leaf in range(clustering.n_leaves_):
            self.children[leaf] = []

        # Store the set of leaves and nodes
        self.leaves = range(clustering.n_leaves_)
        self.nodes = range(clustering.n_leaves_ + len(clustering.children_))

        # Store the label counts of each node
        self.counts = {}
        for node in self.nodes:
            self.counts[node] = {}
            for label in labels:
                self.counts[node][label] = 0

        # Store known labels
        self.known = {}

    def learn(self, point, label):
        '''Learn the label of a data point'''

        self.known[point] = label

        # Update the label counts of all ancestors
        for node in self.ancestors(point):
            self.counts[node][label] += 1


    def ancestors(self, node):
        if node in self.parent:
            return [node] + self.ancestors(self.parent[node])
        else:
            return [node]


    def marginal_likelihood(self, counts):
        '''Returns the marginal likelihood distribution'''
        successes = counts[0]
        trials = counts[0] + counts[1]
        # return quad(lambda probability: binom.pmf(successes, trials, probability), 0, 1)[0]
        return 1 / binom_coefficient(trials, successes) / (trials + 1)


    def predictive(self, counts):
        '''Returns the predictive distribution'''
        successes = counts[0]
        failures = counts[1]
        # Using Bayes-Laplace (uniform) prior rather than Jeffreys prior for speed
        # predictive_success = quad(lambda probability: probability * beta.pdf(probability, 1 + successes, 1 + failures), 0, 1)[0]
        predictive_success = (successes + 1) / (successes + failures + 2)
        return {0: predictive_success, 1: 1 - predictive_success}


    def predict(self):
        '''Predict the labels of the dataset'''

        # print 'Calculating marginal likelihoods...'
        marginal_likelihood = {}
        probability = {}
        for node in self.nodes:
            marginal_likelihood[node] = self.marginal_likelihood(self.counts[node])
            probability[node] = .5 * marginal_likelihood[node] + .5 * np.product([probability[child] for child in self.children[node]])
            
        # print 'Calculating predictives...'
        predictive = {}
        for node in self.nodes:
            predictive[node] = self.predictive(self.counts[node])

        predictions = []
        for leaf in self.leaves:

            if leaf in self.known:
                predictions.append(self.known[leaf])
                continue

            leaf_predictive = {label: 0 for label in self.labels}

            for node in self.ancestors(leaf):

                weight = probability[node] * np.product([(1 - probability[ancestor]) for ancestor in self.ancestors(node)[1:]])

                for label in self.labels:
                    leaf_predictive[label] += predictive[node][label] * weight

            prediction = max(leaf_predictive.keys(), key=lambda label: leaf_predictive[label])

            predictions.append(prediction)


        return predictions







