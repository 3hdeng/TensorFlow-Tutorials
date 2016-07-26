# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt



from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits = load_digits()
print(digits.data.shape)

fig = plt.figure(figsize=(3, 3))

plt.imshow(digits['images'][66], cmap="gray", interpolation='none')

#plt.show()
plt.savefig("lesson9-digits.png")

#=================
from sklearn import svm
classifier = svm.SVC(gamma=0.001)
#classifier.fit(digits.data, digits.target)
#predicted = classifier.predict(digits.data)

import numpy as np
# print(np.mean(digits.target == predicted))

from sklearn.cross_validation import train_test_split
X=digits.data
y=digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("X_train: " , X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print(np.mean(y_test == predicted))


#=========== 
from tensorflow.contrib import skflow
n_classes = len(set(y_train))
print(n_classes)
# classifier = skflow.TensorFlowLinearClassifier(n_classes=n_classes)
# xxx classifier = skflow.TensorFlowDNNClassifier(n_classes=n_classes)
 # Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = skflow.DNNClassifier(hidden_units=[20, 20, 10], n_classes=n_classes)
# optimizer=tf.train.ProximalAdagradOptimizer(
#      learning_rate=0.1,
#      l1_regularization_strength=0.001
#    ))
classifier.fit(X_train, y_train,steps=1000)    

y_pred = classifier.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
print(np.mean(y_test == y_pred))