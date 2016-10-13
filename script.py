import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


# Load the digits dataset
digits = load_digits()


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


# Store the counts of each category for each node
get_counts = {}
for node in range(clustering.n_leaves_ + len(clustering.children_)):
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


root = len(clustering.children_) + clustering.n_leaves_ - 1
print get_counts[root]













