#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# python list has no attr shape
# Use numpy.array to use shape attribute.
input=np.array( [[[1, 1, 1], [2, 2, 2]],
        [[30, 31, 32], [40, 41, 42]],
        [[50, 51, 52], [60, 61, 62]]
        ] )
print(input.shape)        

x=tf.slice(input, [1, 0, 0], [1, 1, 3])
print('tensor x', x)

y=tf.slice(input, [1, 0, 0], [1, 2, 3])
print('tensor y', y)

z=tf.slice(input, [1, 0, 0], [2, 1, 3])
print('tensor z',z)
print("=================")                

                    
model = tf.initialize_all_variables()                
with tf.Session() as sess:
    # you need to initialize all variables
    # tf.initialize_all_variables().run()
    sess.run(model)
    # print(sess.run([x,y,z]) )
    print(sess.run(x))
    print("=================")
    print(sess.run(y))
    print("=================")
    print(sess.run(z))
    print("=================")
    z_val=sess.run(z)
    print(z_val.shape)
    print(z_val[1,0,0])
    print(z_val[1,0,1])
    print(z_val[1,0,2])
  
t=np.array( [30, 31, 32])
print(t.shape)  
#xxx t=np.array( [[30] [31] [32]])
t=np.array([ [30, 31, 32] ])
print(t.shape)    
t0=np.array([ [30, 31, 32],  [40, 41, 42] ])
print(t0.shape)
t1=np.array([[[30, 31, 32],  [40, 41, 42]]])
print(t1.shape)


"""
[[[30 31 32]]]
=================
[[[30 31 32]
  [40 41 42]]]
=================
[[[30 31 32]]

 [[50 51 52]]]
=================
"""