#!/usr/bin/env python
#np.ndarray

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops


 
""" tensor object has no attr shape
arr= np.random.rand(11,28, 28,16)
arr_tensor = tf.Variable(arr, name="arr")
print "arr_tensor"
print type(arr_tensor)
print arr_tensor.shape

tf_a= tf.constant(1.0)
print "tf_a"
print type(tf_a)
print tf_a.shape

tf_b= tf.Variable(False)
print "tf_b"
print type(tf_b)
print tf_b.shape

xxx tf_c= tf.Variable(3.0, (4,3))
tf_c= tf.Variable((4,3))
print "tf_c"
print type(tf_c)
print tf_c.shape

"""

blah = 1
# xxx blah_name = [ k for k,v in locals().iteritems() if v is blah][0]
vdict=locals()
blah_name = [ k for k in vdict.keys() if vdict[k] is blah][0]
print blah_name

tf_a= tf.constant(1.0)   # python float to 0-D tensor
tf_b= tf.Variable(False) # python bool to 0-D tensor
tf_a0= tf.Variable(1.0)
tf_a1= tf.Variable([1.0])
tf_a11= tf.Variable((1.0,))
# xxxx tf_a11= tf.Variable((1.0,))

tf_c= tf.Variable((4,3)) # python tuple to 1-D tensor
tf_d= tf.Variable([4,3]) # python list/vector to 1-D tensor
tf_e= tf.Variable(np.array([4,3])) # numpy array to 1-D tensor
# xxxtf_f= tf.Variable(np.array(4,3)) # numpy array to 1-D tensor
tf_f= tf.Variable(np.ndarray(shape=(4,3), dtype=float, order='F')) #ndarray to 2-D tensor

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

init_op = tf.initialize_all_variables()

with tf.Session() as session:
        session.run(init_op)
        #xxx print l1a_tensor.shape
        a=session.run(tf_a)
        b=session.run(tf_b)
        c=session.run(tf_c)
        d=session.run(tf_d)
        e=session.run(tf_e)
        f=session.run(tf_f)
        print('a', a.shape)
        a0=session.run(tf_a0)
        a1=session.run(tf_a1)
        a11=session.run(tf_a11)
        print('a0', a0.shape, a0)
        print('a1', a1.shape, a1)
        print('a11', a11.shape, a11)
        
        print('b', b.shape, b)
        print('c', c.shape, c)
        print('d', d.shape, d)        
        print('e', e.shape, e)    
        print('f', f.shape, f)      

        
        

