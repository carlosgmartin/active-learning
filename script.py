import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import binom
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
adjusted_rand_score(clustering.labels_, digits.target)






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
for leaf in range(clustering.n_leaves_):
	query(leaf)




# Returns the marginal likelihood (or model evidence) of a binomial distribution
def marginal_likelihood(successes, trials):
	return quad(lambda probability: binom.pmf(successes, trials, probability), 0, 1)[0]




get_likelihood = {}
get_posterior = {}

# Print the marginal likelihood of each node
for node in reversed(traversal):

	trials = sum(get_counts[node].values())
	successes = get_counts[node][0]

	get_likelihood[node] = marginal_likelihood(successes, trials)
	
	get_posterior[node] = .5 * get_likelihood[node] + .5 * np.product([get_posterior[child] for child in get_children[node]])

	print '{:.2f}'.format(get_likelihood[node])
	print '{:.2f}'.format(get_posterior[node])

	print












