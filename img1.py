# How do I read an image file using Python?

import os,sys
import Image
jpgfile = Image.open("picture.jpg")

print jpgfile.bits, jpgfile.size, jpgfile.format