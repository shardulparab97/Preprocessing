import skimage
import skimage.io as io
import skimage.transform
import sys
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.image as mpimg

IMAGE_LOCATION = "input_image.jpg"

#Read Image
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)

#For converting to BGR
imgBGR = img[:, :, (2, 1, 0)]

#For resizing
imgResize = skimage.transform.resize(img, (128, 128))

#For converting HWC to CHW Format
imgCHW	 = imgResize.swapaxes(1, 2).swapaxes(0, 1)

#Add axis of batches
imgFinal = imgCHW[np.newaxis, :, :, :].astype(np.float32)

print("Final shape is:",imgFinal.shape)


