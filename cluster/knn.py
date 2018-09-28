'''
Python implementation of K-nearest neighbours (KNN) classifier using scikit-learn
'''

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.datasets import make_classification

# A random data set
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, n_classes=3, \
             n_clusters_per_class=2, random_state=0, shuffle=False)

#KNN
'''
The training algorithm for the KNN classifier uses some similarity measure (e.g. Euclidean/Manhattan/Minkowski distance functions)
to classify new data instances based on the classifications of its surrounding (nearest) neighbours, as computed by an
appropriate search algorithm (ball tree and kd tree algorithms; or else use a brute force search). The tree search algorithms
utilise the distance function measure to identify the K nearest neighbours. One must also specify a leaf size.
An instance is classified by a majority vote of its neighbours, i.e. the instance is assigned to the class most common amongst its
K nearest neighbours measured by a distance function. Alternatively, weights may be chosen for the neighbours' votes, e.g. based on
some function of inverse distance, so that closer neighbours of a query point will have greater influence than those further away.
One must choose the number of classes and the number K of neighbours considered.
The KNN algorithm is peculiar because it is sensitive to the local structure of the data - e.g. classes may appear as 'islands'.
This is not possible in, for example, k-means clustering
KNN tends to bias predictions towards classes that occur more frequently, simply because they are more likely to appear as one of
the nearest neighbors. This can be remedied by distance weighting or by weighting an instance according to its 'support' (i.e. the
no. of instances with that target label).
N.B. Minkowski distance is a metric suitable for use in high-dimensional space and one must specify a power parameter p for the
Minkowski metric. For categorical variables the Hamming distance must be used. Beware that a mixture of numerical and categorical
variables can mean some variables have a greater influence - one solution is to standardise the variables of the training set.
'''

knn_clf = KNeighborsClassifier(n_neighbors=6,weights='distance',algorithm='ball_tree',p=2,metric='minkowski')
knn_clf.fit(X[:800],y[:800]) # train data

print knn_clf.score(X[800:],y[800:]) # accuracy on test data
