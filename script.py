import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

digits = load_digits()

clustering = AgglomerativeClustering(n_clusters=len(np.unique(digits.target)))
clustering.fit(digits.data)

print adjusted_rand_score(clustering.labels_, digits.target)