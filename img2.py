import tensorflow as tf
import numpy as np
from PIL import Image

imgfname_list=['brain_left_right.jpg']
filename_queue = tf.train.string_input_producer(imgfname_list) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
with sess.as_default():
    sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(len(imgfname_list)): #length of your filename list
  image = my_img.eval() #here is your image Tensor :) 

Image.fromarray(np.asarray(image)).show()

coord.request_stop()
coord.join(threads)