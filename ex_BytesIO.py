# http://stackoverflow.com/questions/26879981/writing-then-reading-in-memory-bytes-bytesio-gives-a-blank-result
"""
I tried writing to a zip file in memory, and then reading the bytes back out of that zip file. 
So instead of passing in a file-object to gzip, I pass in a BytesIO object. 
Here is the entire script:
"""

from io import BytesIO
import gzip

# write bytes to zip file in memory
myio = BytesIO()
g = gzip.GzipFile(fileobj=myio, mode='wb')
g.write(b"does it work")
g.close()

# read bytes from zip file in memory
myio.seek(0)
g = gzip.GzipFile(fileobj=myio, mode='rb')
result = g.read()
g.close()

print(result)
# But it is returning an empty bytes object for result

# need to seek back to the beginning of the file after writing the initial in memory file...
# myio.seek(0)
# np.ndarray.
#  dumps()	Returns the pickle of the array as a string.
#  tobytes([order])	Construct Python bytes containing the raw data bytes in the array.
#  tostring([order])	Construct Python bytes containing the raw data bytes in the array.

# Do the simplest possible python code to load your image, e.g.:
"""
from PIL import Image
im = Image.open('test.png')

# or

from PIL import Image
import io
with open('test.png') as f:
   io = io.BytesIO(f.read())
im = Image.open(io)
"""

from PIL import Image
#from io import BytesIO

# First, load the image again
filename = "brain_left_right.jpg"


#=== non-tf test
ms = BytesIO()
bytes = Image.open(filename) #xxx.tobytes()
bytes.save(ms, format = "JPEG")
ms.flush()
ms.seek(0)
im=Image.open(ms) # <- Error here