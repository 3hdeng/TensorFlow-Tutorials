#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile
from google.protobuf import text_format

trX = np.linspace(-1, 1, 101)
#trX=trX.astype(np.float32)
# [-1,0, 1, 2, ...100]
# trX is a numpy.ndarray of shape (101,)
# trX.shape=(101,), shape : tuple of ints

# xxx print(typeof trX)
# f(arg1,arg2, *argx, **karg)
print(type(trX))

trY = 2.0 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
#trY=trY.astype(np.float32)
print(trX.shape)
print(trY.shape)
print type(trX[0])
print type(trY[0])

#exit()

X = tf.placeholder("float") #dtype=tf.float32)
Y = tf.placeholder("float") #dtype=tf.float32)


def model(X, w):
    return tf.mul(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

# cost = tf.square(Y - y_model) # use square error for cost function
# cost=tf.identity(cost, name="cost")
cost=tf.square(Y- y_model, name="cost")
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()

    for i in range(2): # number of epochs
        for (x, y) in zip(trX, trY): # 101 data points per train_op
            sess.run(train_op, feed_dict={X: x, Y: y})
            # w_val, cost_val=sess.run([w,cost], feed_dict={X: x, Y: y})  
            # print(sess.run(cost))
            # xxx print(w.eval(feed_dict={X:x, Y:y}))
            # w_val = w.eval({X:x,Y:y}, sess)
            w_val=w.eval(sess)
            #print(w_val)
            
    #print(cost_val)
    print(w_val)

    # os.system("rm -rf /tmp/load")
    # tf.train.write_graph(sess.graph_def, "/tmp/load", "test.pb", False) 
    tf.train.write_graph(sess.graph_def,"", "gd1.pbtxt", True)
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess,"chkpt1.data")


with tf.Session() as sess2:
     print("load graph")
     with gfile.FastGFile("gd1.pbtxt",'r') as f:
        graph_def = tf.GraphDef()
        # graph_def.ParseFromString(f.read())
        text_format.Merge(f.read(), graph_def)
        sess2.graph.as_default() # default graph, the current active graph
        tf.import_graph_def(graph_def, name='')

     print("map variables")
     cost2 = sess2.graph.get_tensor_by_name("cost:0")
     w2 = sess2.graph.get_tensor_by_name("weights:0")
     tf.add_to_collection(tf.GraphKeys.VARIABLES, cost2)
     tf.add_to_collection(tf.GraphKeys.VARIABLES, w2)

     try:
         saver = tf.train.Saver(tf.all_variables()) # 'Saver' misnomer! Better: Persister!???
     except:pass
     print("load data")
     saver.restore(sess2, "chkpt1.data")  # now OK
     print(cost2.eval({X:0.8, Y:1.8}, sess2))
     print(w2.eval({X:0.8, Y:1.8}, sess2))
     print("DONE")

