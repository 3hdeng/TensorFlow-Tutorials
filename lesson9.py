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

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print(np.mean(y_test == predicted))