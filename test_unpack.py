import tensorflow as tf
import numpy as np

a=np.array( [[[1, 1, 1], [2, 2, 2]],
        [[30, 31, 32], [40, 41, 42]],
        [[50, 51, 52], [60, 61, 62]]
        ] )
t= tf.convert_to_tensor(a, dtype=tf.int32)
print(t.get_shape())

u= tf.unpack(t) 
# return The list of Tensor objects unpacked from value.
# xxx print(u.get_shape())
print("u= ", u)
print("=====  session run ======")
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(t))
    print(session.run(u))