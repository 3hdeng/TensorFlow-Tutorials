#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops

# input numpy ndarray
def calc_shape(l1a):
    shape0=l1a.shape
    #print 'l1a.shape = {0}'.format(shape0)
    shape1=shape0[1:3] # [?,28,28,16] --> [28,28]
    #print 'l1a.shape trucated to  {0}'.format(shape1)
    shape1=np.append([2], shape1)# --> [1,28,28]
    #print 'l1a.shape trucated and reshaped to  {0}'.format(shape1)
    return shape1
    
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
        #xxx print l1a_tensor.shape
        print(session.run(l1a_tensor).shape)


#=========
def myplot(X_val, L1a_val,i):    
        #====================================
        print 'myplot to save the images' 
        # plot original image and first and second components of output
        fig = pyplot.figure()
        ax = fig.add_subplot(131)
        ax.axis('off');
        pyplot.imshow(X_val) #, cmap='gray')
        
        # recall that the convOp output (filtered image) is actually a "minibatch",
        # of size 1 here, so we take index 0 in the first dimension:
        ax=fig.add_subplot(132); ax.axis('off'); pyplot.imshow(L1a_val[0])#, cmap='gray')
        ax=fig.add_subplot(133); ax.axis('off'); pyplot.imshow(L1a_val[1])#, cmap='gray')
        pyplot.savefig('l1a_{0}.jpg'.format(i) )
        # fig.close()
        
#====================================
#===== use tf.py_func
shape1= calc_shape(l1a)
L1a_sample_tensor=tf.py_func(L1a_sampling, [l1a_tensor],[tf.float64])
with tf.Session() as session:
        session.run(init_op)
        #xxx print l1a_tensor.shape
        result=session.run(L1a_sample_tensor)
        print type(result)
        #print(session.run(L1a_sample_tensor).shape)
        L1a_val=np.reshape(result, shape1)
        print(L1a_val.shape)
        
        
myplot(l1a[3,:,:,0], L1a_val, 0)
