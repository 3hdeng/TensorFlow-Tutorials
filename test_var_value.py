import tensorflow as tf
import numpy as np

a=np.array( [[[1, 1, 1], [2, 2, 2]],
        [[30, 31, 32], [40, 41, 42]],
        [[50, 51, 52], [60, 61, 62]]
        ] )
t= tf.convert_to_tensor(a, dtype=tf.int32)
x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(t))
    
    for i in range(5):
        #session.run(model)
        x = x + 1
        #print(x)
        print(session.run(x))