import numpy as np
import sys
from PIL import Image
import os

inputfile = sys.argv[1]
outputfile = sys.argv[2]

# load image RGB data
array = np.loadtxt(inputfile)
width = int(array[0])
height = int(array[1])
print("width=%d" % (width))
print("heigth=%d" % (height))

# extract rgb content
rgb = np.delete(array, [0, 1], None)
rgb = rgb.reshape(width, height, 3)
rgb = rgb.astype(np.uint8)

# save image
img = Image.fromarray(rgb)
img.save(outputfile)
