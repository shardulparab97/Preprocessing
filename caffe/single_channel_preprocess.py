#code for preprocessing the image for caffe along with converting to grayscale/single channel
import skimage
import skimage.io as io
import skimage.transform
from skimage.color import rgb2gray
import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot
import matplotlib.image as mpimg
import scipy.misc
import cv2

#Taking image input
IMAGE_LOCATION = "input_image.jpg"

#Read Image
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Final shape is:",img.shape)

#converting to grayscale
img = rgb2gray(img)


#For resizing
imgResize = skimage.transform.resize(img, (128, 128))

#adding the axis for channel
imgResize = imgResize[np.newaxis, :, :].astype(np.float32)

imgCHW = imgResize
#Add axis of batches
imgFinal = imgCHW[np.newaxis, :, :, :].astype(np.float32)

print("Final shape is:",imgFinal.shape)