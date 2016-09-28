import tensorflow as tf


#x = tf.Variable(0, name='x')
x= tf.get_variable("x", [100])
# Gets an existing variable with these parameters or create a new one.
init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    print( tf.shape(x) )
    # print(tf.Tensor.get_shape(x))
    # TypeError: unbound method get_shape() must be called with Tensor instance as first argument (got Variable instance instead)
    print(x.get_shape())
    
    x_val=session.run(x)
    print(x_val)
    print(x_val.shape)
  