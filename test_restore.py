import tensorflow as tf

# v3 = tf.Variable("deliberate dummy var", name="v3")
# has to know the var names before restore ?
# xxx v1= tf.Variable(0.01, name="v1")
#  Expected to restore a tensor of type float, got a tensor of type int32 instead: tensor_name = v1
v1= tf.Variable(44, name="v1")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  val=sess.run(v1)
  print(val)
  val=sess.run(biases)
  print(val)


""" 
$ find ./ -name  inspect_checkpoint*
./contrib/learn/python/learn/utils/inspect_checkpoint.py

from tensorflow.contrib.learn.python.learn.utils  import inspect_checkpoint as inspect
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "v1")
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "")
inspect.print_tensors_in_checkpoint_file("/tmp/model.ckpt", "weights")
"""

