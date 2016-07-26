import tensorflow as tf

print range(7)
x = tf.Variable(0., name='x')
threshold = tf.constant(5.)
th1=5.0
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    # while x_value< th1:
    while session.run(tf.less(x, threshold)):
        x = x + 1
        #x_value = session.run(x)
        #print(x_value)