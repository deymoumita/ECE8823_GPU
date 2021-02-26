import numpy as np
import sys
import PIL
from PIL import Image
import os

inputfile = sys.argv[1]
outputfile = sys.argv[2]

# load image
img = Image.open(inputfile)
array = np.array(img)
width = array.shape[0]
height = array.shape[1]
print("width=%d" % (width))
print("heigth=%d" % (height))

# dump image RGB data
with open(outputfile, 'w') as f:
	f.write('%d %d ' % (width, height))
	for y in range(height):
		for x in range(width):
			for c in range(3):
				f.write(' %d' % (array[y, x, c]))
