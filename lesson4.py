import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

# First, load the image again
filename = "brain_left_right.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [200, 0, 0], [100, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()


"""
brain_left_right.jpg: 580x380
tensorflow.python.framework.errors.InvalidArgumentError: Expected begin[0] in [0, 580], but got 1000
         [[Node: Slice = Slice[Index=DT_INT32, T=DT_UINT8, _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_Placeholder_0, Slice/begin, Slice/size)]]
Caused by op u'Slice', defined at:
  File "lesson4.py", line 10, in <module>
    slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])
"""