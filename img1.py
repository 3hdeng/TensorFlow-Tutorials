# How do I read an image file using Python?

import os,sys
# import Image
from PIL import Image


jpgfile = Image.open("brain_left_right.jpg")

print jpgfile.bits, jpgfile.size, jpgfile.format