# lesson9, exercise
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets, cross_validation, metrics
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn import monitors
from tensorflow.contrib import skflow


# Load dataset
def unpickle(file): 
    import cPickle 
    fo = open(file, 'rb') 
    dict = cPickle.load(fo) 
    fo.close() 
    return dict


def load_cifar(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as inf:
        cifar = pickle.load(inf) #, encoding='latin1')
    data = cifar['data'].reshape((10000, 3, 32, 32))
    data = np.rollaxis(data, 3, 1)
    data = np.rollaxis(data, 3, 1)
    y = np.array(cifar['labels'])
    # Just get 2s versus 9s to start
    # Remove these lines when you want to build a big model
    # mask = (y == 2) | (y == 9)
    # data = data[mask]
    # y = y[mask]
    return data, y

X_trains=[]
y_trains=[]
X_vals=[]
y_vals=[]

for i in range(1,6):
    X,y= load_cifar("cifar-10-batches-py/data_batch_" + str(i))    
    # Split it into train / test subsets
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)

    # Split X_train again to create validation data
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X,
                                                                   y,
                                                                   test_size=0.2,
                                                                   random_state=42)

    X_trains.append(X_train)
    y_trains.append(y_train)
    X_vals.append(X_val)
    y_vals.append(y_val)

# TensorFlow model using Scikit Flow ops
X_test,y_test= load_cifar("cifar-10-batches-py/test_batch") 

n_classes=10

# http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/

def conv_model(X, y):
    X = tf.expand_dims(X, 3)
    features = tf.reduce_max(learn.ops.conv2d(X, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    return learn.models.logistic_regression(features, y)


# Create a classifier, train and predict.
classifier = learn.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                        steps=100, learning_rate=0.05,
                                        batch_size=200)
for i in range(0,5):
    val_monitor = monitors.ValidationMonitor(X_vals[i], y_vals[i], every_n_steps=50)
    classifier.fit(x=X_trains[i], y=y_trains[i], monitors=[val_monitor])
    
    
    
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Test Accuracy: {0:f}'.format(score))