#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops

# input numpy ndarray
def L1a_sampling(l1a):
    """
    shape0=l1a.shape
    print 'l1a.shape = {0}'.format(shape0)
    shape1=shape0[1:3] # [?,28,28,16] --> [28,28]
    print 'l1a.shape trucated to  {0}'.format(shape1)
    shape1=np.append([1], shape1)# --> [1,28,28]
    print 'l1a.shape trucated and reshaped to  {0}'.format(shape1)
    """
    l1a_0=l1a[0,:,:,0] # the 0-th output feature for the 0th input, [28,28]
    #print 'l1a sample0 shape = {0}'.format(l1a_0.shape)
    l1a_15=l1a[0,:,:,15] # the 15-th output feature for the 0th input
    #l1a_sample=tf.concat(0, [l1a_0, l1a_15]) # concate at dim 0
    return np.append([l1a_0], [l1a_15], 0)

l1a= np.random.rand(11,28, 28,16)
L1a_sample= L1a_sampling(l1a)

print L1a_sample.shape


l1a_tensor = tf.Variable(l1a, name="l1a")
init_op = tf.initialize_all_variables()

with tf.Session() as session:
        session.run(init_op)
        print l1a_tensor.shape
        #print(session.run(x))

