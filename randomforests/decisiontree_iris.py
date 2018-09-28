from sklearn.tree import DecisionTreeClassifier
import numpy as np
import unveil_tree_structure

# Iris dataset in fmt: sepal length / sepal width / petal length / petal width / species
iris_data = np.genfromtxt("iris.csv", \
                dtype=[("sep_len",float),("sep_wid",float),("pet_len",float),("pet_wid",float),("species","|S10")], \
                delimiter=",",skip_header=1)
'''
print type(iris_data) # a ndarray, as usual
quack = iris_data[0]
print iris_data["sep_len"] # you can access columns in the table by their names
print type(quack) # note how the type of an individual row in the array is 'void' (used when there are mixed data types)
print quack["sep_len"]
'''

traindata = np.array(iris_data.tolist())
traindata = np.array(traindata[:,:-1],dtype=float)

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=17) # Create a binary decision tree with a max of 2 decisions
             # uses the CART algorithm with the (default) Gini impurity in the cost function for cutting the tree
# Create the decision tree based on parameters of petal length & petal width
tree_clf.fit(traindata[:,2:],iris_data[:]["species"])
print "Gini importances of features:\n", tree_clf.feature_importances_
unveil_tree_structure.get_tree_info(tree_clf)

expected_output = []
species_dict = {"setosa": 1, "versicolor": 3, "virginica": 4} # dict for species to expected output nodes
for instance in iris_data["species"]:
    expected_output.append(species_dict[instance])
unveil_tree_structure.get_output_info(tree_clf.apply(traindata[:,2:]), expected_output)

# Estimate class probabilities and make predictions for new instances
print tree_clf.predict_proba([[5,1.5]])
print tree_clf.predict([[5,1.5]])

print tree_clf.classes_
