from __future__ import division
import numpy as np
from scipy.special import binom
from sklearn.cluster import AgglomerativeClustering


class Learner:

    def __init__(self, data, labels):
        ''' Initialize the learner '''

        # Store the set of possible labels
        self.labels = labels

        # Create a hierarchical clustering
        clustering = AgglomerativeClustering()

        # Fit the clustering to the data
        clustering.fit(data)

        # Define the set of leafs and nodes
        self.leafs = range(clustering.n_leaves_)
        self.nodes = range(clustering.n_leaves_ + len(clustering.children_))

        # Store the parent and children of each node
        self.parent = {}
        self.children = {}
        for parent, children in enumerate(clustering.children_, clustering.n_leaves_):
            self.children[parent] = children
            for child in children:
                self.parent[child] = parent
        for leaf in self.leafs:
            self.children[leaf] = []

        # Store the ancestors of each node
        self.ancestors = {}
        for node in reversed(self.nodes):
            if node not in self.parent:
                self.ancestors[node] = [node]
            else:
                self.ancestors[node] = [node] + self.ancestors[self.parent[node]]

        self.total = {}
        self.count = {}
        self.predictive = {}
        self.evidence = {}
        self.weight = {}
        self.leaf_predictive = {}
        self.known = {}

        # Set counts of each leaf
        for leaf in self.leafs:
            self.total[leaf] = 0
            for label in self.labels:
                self.count[leaf, label] = 0

        # Update all nodes
        self.update(self.nodes)



    def query(self):
        ''' Choose a leaf to query '''

        return self.query_random()



    def divergence(self, prior, posterior):
        ''' Returns the KL divergence between two categorical distributions '''
        return sum(posterior[label] * np.log(posterior[label] / prior[label]) if posterior[label] != 0 else 0 for label in self.labels)



    def query_active(self):
        # Find the leafs that have not been labeled yet
        remaining = list(set(self.leafs).difference(self.known.keys()))
        
        # Estimate the value of querying each leaf (expected KL divergence)
        value = {}
        for leaf in remaining:
            # Prior distribution
            prior = {label: self.leaf_predictive[leaf, label] for label in self.labels}

            value[leaf] = 0

            for label in self.labels:
                # Probability that this point has this label
                probability = self.leaf_predictive[leaf, label]

                # Posterior distribution if this point were labeled with this label
                posterior = {other_label: 1 if other_label == label else 0 for other_label in self.labels}

                # KL divergence
                divergence = self.divergence(prior, posterior)

                # Expected KL divergence
                value[leaf] += probability * divergence

        # Choose the leaf with the highest value
        return max(value.keys(), key=lambda leaf: value[leaf])





    def query_random(self):
        # Find the leafs that have not been labeled yet
        remaining = list(set(self.leafs).difference(self.known.keys()))
        
        return np.random.choice(remaining)



    def query_max_expected(self):

        # Find the leafs that have not been labeled yet
        remaining = list(set(self.leafs).difference(self.known.keys()))
        
        # Use a subset of these
        remaining = np.random.choice(remaining, size=50, replace=False)

        # Estimate the value of querying each leaf
        value = {}

        for leaf in remaining:
            value[leaf] = 0
            
            # Expected number of correct labels given the label of this point
            for label in self.labels:

                # Probability that this point has this label
                probability = self.leaf_predictive[leaf, label]
                
                # Expected number of correct labels given that this point has this label
                self.learn(leaf, label)
                value[leaf] += probability * self.expected_correct()
                self.unlearn(leaf, label)
        
            # print 'Value of leaf {}: {:.0f}'.format(leaf, value[leaf])

        # Choose the leaf that maximizes the expected number of correct labels
        return max(value.keys(), key=lambda leaf: value[leaf])


    def update(self, path):
        ''' Update a path of nodes '''

        posterior = {}

        # Bottom-up pass
        for node in path:

            # Update counts
            if node not in self.leafs:
                self.total[node] = sum(self.total[child] for child in self.children[node])
                for label in self.labels:
                    self.count[node, label] = sum(self.count[child, label] for child in self.children[node])

            # Update predictives
            for label in self.labels:
                self.predictive[node, label] = (self.count[node, label] + 1) / (self.total[node] + len(self.labels))

            # Update the likelihood
            likelihood = 1 / (binom(self.total[node], self.count[node, 0]) * (self.total[node] + 1))
            
            # Set the prior
            prior = .5

            # Update the joint probability
            joint = prior * likelihood

            # Update evidence
            self.evidence[node] = joint + (1 - prior) * np.product([self.evidence[child] for child in self.children[node]])

            # Update posterior
            posterior[node] = joint / self.evidence[node]

        # Top-down pass
        for node in reversed(path):

            # Update weight
            self.weight[node] = posterior[node] * np.product([(1 - posterior[ancestor]) for ancestor in self.ancestors[node][1:]])

            # Faster but more floating point errors
            '''
            if node in self.parent:
                self.weight[node] = posterior[node] * (1/posterior[self.parent[node]] - 1) * self.weight[self.parent[node]]
            else:
                self.weight[node] = posterior[node]
            '''

        # Update leaf predictives
        for leaf in self.leafs:
            for label in self.labels:
                if leaf in self.known:
                    self.leaf_predictive[leaf, label] = 1 if (label == self.known[leaf]) else 0
                else:
                    self.leaf_predictive[leaf, label] = sum(self.predictive[node, label] * self.weight[node] for node in self.ancestors[leaf])
        

    def learn(self, leaf, label):
        ''' Learn the label of a leaf '''

        self.known[leaf] = label

        self.total[leaf] += 1
        self.count[leaf, label] += 1

        # Update all ancestors of this leaf
        self.update(self.ancestors[leaf])


    def unlearn(self, leaf, label):
        ''' Unlearn the label of a leaf '''

        del self.known[leaf]

        self.total[leaf] -= 1
        self.count[leaf, label] -= 1

        # Update all ancestors of this leaf
        self.update(self.ancestors[leaf])


    def predict(self, leaf):
        ''' Predict the label of a leaf '''
        return max(self.labels, key=lambda label: self.leaf_predictive[leaf, label])


    def expected_correct(self):
        ''' Expected number of correct labels '''
        return sum(max(self.leaf_predictive[leaf, label] for label in self.labels) for leaf in self.leafs)








