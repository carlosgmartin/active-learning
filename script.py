from __future__ import division
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import binom, beta
from scipy.integrate import quad
from collections import deque


# Load the digits dataset
digits = load_digits()

# Make this a binary classification problem
digits.target = digits.target < 5



# Find the target categories
categories = np.unique(digits.target)


# Create a hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=len(categories))


# Fit the hierarchical clustering on the data
clustering.fit(digits.data)

# Measure similarity between predicted and target clusterings
print 'Rand index: {:.2f}'.format(adjusted_rand_score(clustering.labels_, digits.target))






# Store the parent and children of each node
get_parent = {}
get_children = {}
for parent, children in enumerate(clustering.children_, clustering.n_leaves_):
    get_children[parent] = children
    for child in children:
        get_parent[child] = parent

for leaf in range(clustering.n_leaves_):
    get_children[leaf] = []



# Returns the traversal of a tree
def traverse(node):
    traversal = []

    deq = deque()
    deq.append(node)
    while len(deq) > 0:
        node = deq.popleft() # pop = DFS, popleft = BFS
        traversal.append(node)
        for child in get_children[node]:
            deq.append(child)

    return traversal

# Find the root of the tree
root = len(clustering.children_) + clustering.n_leaves_ - 1

# Find the tree traversal
traversal = traverse(root)



# Store the counts of each category for each node
get_counts = {}
for node in traversal:
    get_counts[node] = {}
    for category in categories:
        get_counts[node][category] = 0




def query(leaf):
    assert leaf < clustering.n_leaves_

    # Find the category corresponding to this leaf
    category = digits.target[leaf]

    # Increment count of this category for all ancestor nodes
    node = leaf
    while True:
        get_counts[node][category] += 1
        if node in get_parent:
            node = get_parent[node]
        else:
            break


# Query the category of each leaf
for leaf in range(clustering.n_leaves_ // 16):
    query(leaf)








# Returns the evidence (marginal likelihood) distribution
def evidence(observations):
    successes = observations[True]
    trials = observations[True] + observations[False]
    return quad(lambda probability: binom.pmf(successes, trials, probability), 0, 1)[0]

# Returns the predictive distribution
def predictive(observations):
    successes = observations[True]
    failures = observations[False]
    predictive_success = quad(lambda probability: probability * beta.pdf(probability, .5 + successes, .5 + failures), 0, 1)[0] # Using Jeffreys prior (.5, .5)
    return {True: predictive_success, False: 1 - predictive_success}





# Store the merge probability and total probability of each node
get_merge_prob = {}
get_total_prob = {}
for node in reversed(traversal):
    get_merge_prob[node] = evidence(get_counts[node])
    get_total_prob[node] = .5 * get_merge_prob[node] + .5 * np.product([get_total_prob[child] for child in get_children[node]])
    if node in range(clustering.n_leaves_):
        get_total_prob[node] = get_merge_prob[node]


print 'Calculating predictives...'
# Store the predictives of each node
get_predictive = {}
for node in traversal:
    # print '\t{}'.format(node)
    get_predictive[node] = predictive(get_counts[node])

    print get_predictive[node]



# Get the ancestors of a node
def get_ancestors(node):
    if node in get_parent:
        parent = get_parent[node]
        return [parent] + get_ancestors(parent)
    else:
        return []



correct = 0

print 'Making predictions...'

for leaf in range(clustering.n_leaves_):

    total_predictive = {category: 0 for category in categories}
    total_prob = 0
    for ancestor in [leaf] + get_ancestors(leaf):

        total_prob += get_total_prob[ancestor]

        for category in categories:
            total_predictive[category] += get_predictive[ancestor][category] * get_total_prob[ancestor]

    predicted = max(categories, key=lambda category: total_predictive[category])
    target = digits.target[leaf]
    
    correct += (predicted == target)


print correct / len(digits.target)




















