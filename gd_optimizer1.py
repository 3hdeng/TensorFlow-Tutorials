# a basic example for Optimizer

import tensorflow as tf
import numpy as np

print np.random.rand()
print np.random.rand()
print np.random.rand(3)
# xxx print np.random.rand([2,5])
print np.random.rand(2,5)

# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name="w")
# Our model of y = a*x + b
y_model = tf.mul(x, w[0]) + w[1]

# Our error is defined as the square of the differences
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        # session.run(train_op, feed_dict={x: x_value, y: y_value})
        _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
        
    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

"""
The major line of interest here is train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error) 
where the training step is defined. 
It aims to minimise the value of the error Variable, which is defined earlier as the square
of the differences (a common error function). 
The 0.01 is the step it takes to try learn a better value.

//===
We do this in a single line, so that the error is computed only once. 
 _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
 
If we did this is separate lines, it would compute the error, and then the training step, 
it would recompute the error ...

//===
GradientDescentOptimizer
AdagradOptimizer
MomentumOptimizer
AdamOptimizer
FtrlOptimizer
RMSPropOptimizer

Other optimisation methods are likely to appear in future releases of TensorFlow, or in third-party code. 

If you aren’t sure which one to use, use GradientDescentOptimizer unless that is failing.

"""