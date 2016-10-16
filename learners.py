from __future__ import division
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import binom, beta
from scipy.integrate import quad
from scipy.special import binom as binom_coefficient
import numpy as np
from copy import deepcopy


class Learner:


    def choose(self):
        '''Choose a point to label'''
        return self.choose_smart()


    def choose_smart(self):
        '''Use an active learning strategy'''

        # Find the set of points that have not been labeled
        remaining = list(set(self.leaves).difference(self.known.keys()))
        
        # Use a subset of these
        remaining = np.random.choice(remaining, size=50, replace=False)

        # Calculate the estimated value of querying each unlabeled point
        leaf_values = {}

        for leaf in remaining:

            leaf_value = 0

            for outcome in self.labels:
                p_outcome = self.leaf_predictive[leaf][outcome]

                clone = deepcopy(self)
                clone.learn(leaf, outcome)

                # clone.update_leaf_predictive(leaf) # Update leaf predictive distributions
                # This won't work, need to update everything else

                clone.predict() # Update everything
                confidence = clone.confidence()

                leaf_value += p_outcome * confidence

            leaf_values[leaf] = leaf_value
            print 'Value of querying leaf {}:\t{:.4f}'.format(leaf, leaf_value)

        # Choose the leaf with the highest expected query value
        return max(remaining, key=lambda leaf: leaf_values[leaf])


    def choose_random(self):
        '''Use a random sampling strategy'''

        # Find the set of points that have not been labeled
        remaining = list(set(self.leaves).difference(self.known.keys()))
        
        # Choose a random point from this set
        return np.random.choice(remaining)


    def confidence(self):
        '''Returns the expected fraction of correct labels'''
        total = 0
        for leaf in self.leaves:
            if leaf in self.known:
                total += 1
            else:
                total += max(self.leaf_predictive[leaf].values())
        return total / len(self.leaves)


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

        # Store predictives of each leaf
        self.leaf_predictive = {}
        for leaf in self.leaves:
            self.leaf_predictive[leaf] = {}
            for label in self.labels:
                self.leaf_predictive[leaf][label] = 1 / len(self.labels)


        # Make sure everything has been built (need to clean up and restructure this)
        self.predict()

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


    def predictive_dist(self, counts):
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
        self.probability = {}
        for node in self.nodes:
            marginal_likelihood[node] = self.marginal_likelihood(self.counts[node])
            self.probability[node] = .5 * marginal_likelihood[node] + .5 * np.product([self.probability[child] for child in self.children[node]])
        
        # print 'Calculating predictives...'
        self.predictive = {}
        for node in self.nodes:
            self.predictive[node] = self.predictive_dist(self.counts[node])

        predictions = []
        for leaf in self.leaves:

            if leaf in self.known:
                predictions.append(self.known[leaf])
                continue

            # Update the predictive distribution of this leaf
            self.update_leaf_predictive(leaf)

            prediction = max(self.leaf_predictive[leaf].keys(), key=lambda label: self.leaf_predictive[leaf][label])

            predictions.append(prediction)

        return predictions


    def update_leaf_predictive(self, leaf):
        '''Update the predictive distribution of a leaf'''
        self.leaf_predictive[leaf] = {label: 0 for label in self.labels}

        for node in self.ancestors(leaf):

            weight = self.probability[node] * np.product([(1 - self.probability[ancestor]) for ancestor in self.ancestors(node)[1:]])

            for label in self.labels:
                self.leaf_predictive[leaf][label] += self.predictive[node][label] * weight




