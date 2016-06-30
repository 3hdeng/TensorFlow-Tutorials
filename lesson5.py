import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

#=== step1
#x = tf.constant(list(range(10)))
#print(x.eval())

#=== step2
# X = tf.constant(np.eye(10000))
# Y = tf.constant(np.random.randn(10000, 300))

# Z = tf.matmul(X, Y)


#=======exercise========
arr1=np.random.randint(5, size=10000000)
arr2=tf.to_float(arr1, name='ToFloat')

print(arr1.size)
# print(arr2.size)
#'Tensor' object has no attribute 'size'
# print(arr2.shape)
#'Tensor' object has no attribute 'shape'
print arr2
x=arr2.eval()
print(x.shape)


#===

import resource
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()