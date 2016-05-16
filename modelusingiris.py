import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn import datasets
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier
import sys
import logging

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)
train_set_size = 200
X_train = X[:train_set_size]
X_test = X[train_set_size:]
y_train = y[:train_set_size]
y_test = y[train_set_size:]
nn = Classifier(
    layers = [
    Layer("Rectifier", units=100),
    Layer("Softmax")],    
    n_iter=1,
    learning_rule = 'rmsprop')
#w_train = np.array((X_train.shape[0],))
#w_train = []
#w_train[y_train == 0] = 1.2
#w_train[y_train == 1] = 0.8

#nn.fit(X_train, y_train, w_train)
nn.fit(X_train, y_train)

#y_valid = nn.predict(X_valid)
nn.get_parameters()
nn.predict(X_train)
#score = nn.score(X_test, y_test)
#plt.scatter(X_train, y_train,  color='black')
#plt.scatter(X_test, y_test)
#plt.show()
#plt.scatter()
#logging.basicConfig(
 #           format="%(message)s",
  #          level=logging.DEBUG,
   #         stream=sys.stdout)
