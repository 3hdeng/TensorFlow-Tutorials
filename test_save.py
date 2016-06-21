import tensorflow as tf


# Create some variables.
v1 = tf.Variable(3, name="v1")
v2 = tf.Variable(0.1, name="v2")
weights = tf.Variable(tf.random_normal([7, 2], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([10]), name="biases")
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  v1=v1+1
  v2=v2-1 
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
""" 
$ find ./ -name  inspect_checkpoint*
./contrib/learn/python/learn/utils/inspect_checkpoint.py
"""

from tensorflow.contrib.learn.python.learn.utils  import inspect_checkpoint as inspect
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "v1")
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "")
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "weights")

