import tensorflow as tf


x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        x = x + 1
        print(session.run(x))