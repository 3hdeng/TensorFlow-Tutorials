import tensorflow as tf
def read_stl10(filename_queue):
  class STL10Record(object):
    pass
  result = STL10Record()

  result.height = 96
  result.width = 96
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  record_bytes = image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  print value
  record_bytes = tf.decode_raw(value, tf.uint8)

  depth_major = tf.reshape(tf.slice(record_bytes, [0], [image_bytes]),
                       [result.depth, result.height, result.width])
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result
# probably a hack since I should've provided a string tensor

filename_queue = tf.train.string_input_producer(['./data/train_X'])
image = read_stl10(filename_queue)

print image.uint8image
with tf.Session() as sess:
  result = sess.run(image.uint8image)
  print result, type(result)
  
  
  
""" http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
 to initialize the queue runners. The only required thing to add to the code above is the second line from below:

...
with tf.Session() as sess:
    tf.train.start_queue_runners(sess=sess)
...
Afterwards, the image in the result can be displayed with matplotlib.pyplot.imshow(result)
"""