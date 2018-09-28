'''
Python script to implement support vector machine (SVM) using scikit-learn
'''

import numpy as np
from sklearn.svm import SVC # support vector classification
from sklearn.datasets import load_iris

#Iris data set
iris = load_iris()
X = iris.data
y = iris.target

np.random.seed(19)
Z = zip(X,y)
np.random.shuffle(Z)
X,y = zip(*Z)
X_train, y_train = X[:120], y[:120]
X_test, y_test = X[120:], y[120:]

'''
SVMs are supervised learning models with associated learning algorithms that analyse data for classification and regression analysis.
An SVM model is a representation of the samples as points in space, mapped so that the examples fo the separate categories are
divided by a clear gap that is as wide as possible. New examples are mapped into the same space and predicted to belong to a
category based on which side of the gap they fall. In addition to performing linear classification, SVMs can efficiently perform a
non-linear classification using the KERNEL TRICK, implicitly mapping their inputs into high-dimensional feature spaces.
Advtanges :
	1. Effective in high-dimensional spaces
	2. still effective in cases where the number of dimensions is greater than the number of samples
	3. uses only a subset of training points ('support vectors') in the decision functions
	4. versatile through the specification of different kernel functions
Disadvantages:
	1. If the number of features is much greater than the number of samples, avoid over-fitting in choosing kernel functions and
	   and ensure use of regularisation
	2. SVMs do not directly provide probability estimates, these are calculated using an expensive cross-validation procedure

Consider that we have a set x of n data points x_i. In SVMs, a data-point is viewed as a p-dimensional vector x_i, and we want to 
know whether we can separate such points with a (p-1)- dimensional hyperplane. Then we can make the binary classification of data
into two sets, y_i = +/- 1. This is called a LINEAR CLASSIFIER. Any hyperplane is defined by a normal vector w that satisfies:
dot(w,x) - b = 0 ; where b/magn(w) defines the offset of the hyperplane from the origin along the normal vector w.
One reasonable choice for the 'best' hyperplane is that which represents the largest separation (or 'margin') between the two classes.
If the data is linearly separable, we can use hard-margin classification. This corresponds to an optimisation problem of
*minimising* magn(w) subject to y_i*(dot(w,x_i)-b)>=1. (remember that w is the *normal* vector). Notice that the hyperplane is
completely determined by those x_i which lie nearest to it. These x_i are the 'support vectors'.
To extend SVM to cases in which the data are not linearly separable, we minimise a quantity based on a hinge loss function:
(1/n)sum_n(max(0,1-y_i*(dot(w,x_i)-b))) + lambda*(magn(w)**2)
where the parameter lambda determines the tradeoff between increasing the margin size and ensuring that the x_i lie on the correct
side of the margin. For sufficiently small values of lambda, the soft-margin SVM will behave identically to the hard-margin SVM if
the input data are linearly classifiable, but will still learn if a classification rule is viable or not. High values of lambda
may lead to overfitting.
More modern methods are sub-gradient and coordinate descent.

It often happens that the sets to be discriminated are not linearly separable in the space on which the problem is defined. Therefore
the strategy is to map the original space onto a much higher-dimensional space (thereby presumably making the separation easier in
that space). This is achieved using a KERNEL FUNCTION to transform the x_i. A kernel K(a,b) is a function capable of computing the
dot product dot(phi(a).T,phi(b)) based only on the original vectors a and b, without having to compute (or even have knowledge of)
the transformation function phi.
The vector w defining the hyperplane may be written in terms of phi, and dot products with w (i.e. dot(w,phi(x))) for classification
can again be computed by the kernel trick. This allows us to learn a nonlinear classification rule which corresponds to a linear
classification rule for the transformed data points phi(x_i). We do this by writing (in analogy to the usual expression):
b = dot(w,phi(x_i)) - y_i
in terms of the kernel function:
b = sum_{k=1,n} c_k*y_k*K(x_k,x_i) - y_i
So that the new points z can be classified by computing: sgn(dot(w,phi(z)-b)) = sgn([sum_{i=1,n} c_i*y_i*k(x_i,z)] -b)

Every dot product is replaced by a (possibly nonlinear) kernel function. Possible choices besides linear include polynomial, Gaussian
RBF, sigmoid and tanh. The algorithm fits the hyperplane in a transformed feature space. The transformation may be nonlinear and the
transformed space high-dimensional; although the classifier is a hyperplane in the transformed feature space, it may be nonlinear
in the original input space.

SVMs are only directly applicable for binary-class tasks. SVM methods can be extended to multiclassification by using algorithms
that reduce the multi-class task to several binary problems, e.g. implementing the "one-against-one" or "one-against-rest" approach
for the (shape of the) decision function
'''

# linear SV classifier employing a cubic polynomial kernel function, with coeff gamma and no offset (coef0). All class weights are
# equal. The penalty param C of the error term is set to the default of 1.0. This param controls the amount of regularisation
# (larger value of C -> less regularised). The shrinking heuristic is used to help training become faster.
# Probability estimates are enabled (N.B. this slows down the method)
linear_svc = SVC(C=1.0,kernel='poly',degree=3,gamma=0.2,coef0=0.0,tol=1.0E-04,max_iter=-1,decision_function_shape='ovo', \
                 random_state=0,shrinking=True,probability=True)
linear_svc.fit(X_train,y_train)
print "Unique classes are:\n", linear_svc.classes_
print "Number of support vectors for each class:\n", linear_svc.n_support_
print "The support vectors are:\n", linear_svc.support_vectors_
print "Accuracy on test data:\n", linear_svc.score(X_test,y_test)
print "Prediction for a new instance:\nclass:", linear_svc.predict([[5.5,1.7,3.2,1.0]]), "probability", \
        linear_svc.predict_proba([[5.5,1.7,3.2,1.0]])
