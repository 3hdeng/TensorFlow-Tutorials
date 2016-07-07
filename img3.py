import tensorflow as tf

filenames = ['brain_left_right.jpg']
filename_queue = tf.train.string_input_producer(filenames)


reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

images = tf.image.decode_jpeg(value, channels=3)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:  #sess.as_default():
  sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for i in range(len(filenames)): #length of your filename list
  image = images.eval(session=sess) #here is your image Tensor :) 

print(image.shape)
Image.show(Image.fromarray(np.asarray(image)))

coord.request_stop()
coord.join(threads)