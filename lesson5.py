import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

#=== step1
#x = tf.constant(list(range(10)))
#print(x.eval())

#=== step2
# X = tf.constant(np.eye(10000))
# Y = tf.constant(np.random.randn(10000, 300))

# Z = tf.matmul(X, Y)


#=======exercise========
arr1=np.random.randint(256, size=1000000)
arr2=tf.to_float(arr1, name='ToFloat')

print(arr1.size)
# print(arr2.size)
#'Tensor' object has no attribute 'size'
# print(arr2.shape)
#'Tensor' object has no attribute 'shape'
print arr2
x=arr2.eval()
print(x.shape)


#===

import resource
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

session.close()

#=====  exercise

from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg

# First, load the image again
filename = "brain_left_right.jpg"
raw_image_data = Image.open(filename)
#raw_image_data = mpimg.imread(filename)

#===
image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [200, 0, 0], [100, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)


# read data from string
print(type(result))

bytes = result.tobytes()
print(type(bytes))
print(bytes[0:10])

im=Image.frombuffer('RGB', result.shape, bytes)
ms = BytesIO()
im.save(ms, format = "JPEG")
ms.flush()
ms.seek(0)


# readinto(b)
# Read up to len(b) bytes into b, and return the number of bytes read. 
# The object b should be a pre-allocated, writable array of bytes, either bytearray or memoryview.
# arr=bytearray(len(bytes))
# ms.readinto(arr)


# print(arr[0:10])

im = Image.open(ms)
# try Image.open(open("path/to/file", 'rb'))

#im




# ==== log
"""
 $ python lesson5.py
1000000
Tensor("ToFloat:0", shape=(1000000,), dtype=float32)
(1000000,)
107212 Kb
Exception AssertionError: AssertionError("Nesting violated for default stack of <type 'weakref'> objects",) in <bound method InteractiveSession.__del__ of <tensorflow.python.client.session.InteractiveSession object at 0x7fc20eeb6450>> ignored
(100, 550, 3)
Traceback (most recent call last):
  File "lesson5.py", line 55, in <module>
    im = Image.open(BytesIO(result))
  File "/usr/local/lib/python2.7/dist-packages/PIL/Image.py", line 2317, in open
    % (filename if filename else fp))
IOError: cannot identify image file <_io.BytesIO object at 0x7fc20c139b30>
"""