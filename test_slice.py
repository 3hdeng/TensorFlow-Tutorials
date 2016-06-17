#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# python list has no attr shape
# Use numpy.array to use shape attribute.
input=np.array( [[[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]
        ] )
print(input.shape)        

x=tf.slice(input, [1, 0, 0], [1, 1, 3])
print(x)

y=tf.slice(input, [1, 0, 0], [1, 2, 3])
print(y)

z=tf.slice(input, [1, 0, 0], [2, 1, 3])
print(z)
                                    