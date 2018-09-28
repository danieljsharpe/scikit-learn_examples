'''
Python implementation of k-means clustering algorithm using scikit-learn
'''

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.datasets import load_wine

np.random.seed(19)

'''
# random data set
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, n_classes=3, \
             n_clusters_per_class=2, random_state=0, shuffle=False)

print X[:10]
Z = zip(X,y)
np.random.shuffle(Z)
X, y = zip(*Z)
print X[:10]

X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]
'''

#wine data set
X, y = load_wine(True)
Z = zip(X,y)
np.random.shuffle(Z)
X, y = zip(*Z)
X_train, y_train = X[:150], y[:150]
X_test, y_test = X[150:], y[150:]

'''
k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest
mean, such that the data space is partitioned into Voronoi cells defined by means of each cluster as the seed sites.
More formally, the algorithm partitions n observations into k sets so as to minimise the within-cluster sum of squres (WCSS). (i.e.
variance). This is done by iterative refinement using Lloyd's algorithm:
	0. initialise (randomly generate initial 'means' within the data domain)
	1. assign each observation to the cluster whose mean has the least squared Euclidean distance
	2. calculate the new means to be the centroids of the observations in the new clusters
        3. rpt. 1 & 2 until convergence (as defined by a tolerance / max. no. of iterations)
Note that the clustering problem is NP-hard and has a tendency to become stuck in local optima. It is common to run the algorithm
multiple times with different starting conditions to test for robustness. This is controlled by the 'n_init' keyword. The final
result is the best output of n_init consecutive runs in terms of inertia (sum of squared distances of samples to their closest
cluster center). Note that for some cases of initial conditions k-means can be very slow to converge.
An algorithm ("k-means++") has been developed for selection of initial 'means' that aims to avoid the sometimes poor clusterings
found. Else the choise can be made randomly.
The assignment step is also referred to as an 'expectation' step and the update step a 'maximisation' step, making this algorithm
a variant of the generalised expectation-maximisation (EM) algorithm
Note that it is important to run diagnostic checks for determining the number of clusters in the data set, else the clustering
may yield poor results.
'''

kmc_clf = KMeans(n_clusters=3, n_init=10, max_iter=10000, tol=1.0E-05, random_state=0, algorithm="full")
kmc_clf.fit(X_test, y_test)

def calc_acc(res, tgt):
    correct = 0
    for i in range(len(res)):
        if int(res[i]) == int(tgt[i]):
            correct += 1
    return float(correct)/float(len(res))

print "Coords of cluster centres:\n", kmc_clf.cluster_centers_
print "Inertia for final centroids:\n", kmc_clf.inertia_
print "Accuracy on training data:\n", calc_acc(kmc_clf.labels_,y_train)
print "Accuracy on test data:\n", calc_acc(kmc_clf.predict(X_test),y_test)
