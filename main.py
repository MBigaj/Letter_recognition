import sys
import numpy
import numpy as np
from PIL import Image, ImageOps
import os

from plotting_func import plot_all, plot_image, plot_two_images
from convolution_func import tanh, sigmoid, relu, convolve, calc_target_size
from pooling_func import return_pools, max_pooling
from CNN import CNN


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(f'{folder}/{filename}')
        img = ImageOps.grayscale(img)
        img = img.resize(size=(28, 28))
        img = np.array(img)
        img = img.astype(float)
        images.append(img)
    return images


# PRESET FILTERS

# sharpen = np.array([
#     [0, -1, 0],
#     [-1, 5, -1],
#     [0, -1, 0]
# ])
#
# blur = np.array([
#     [0.0625, 0.125, 0.0625],
#     [0.125,  0.25,  0.125],
#     [0.0625, 0.125, 0.0625]
# ])
#
# outline = np.array([
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1]
# ])

# CREATE A FUNCTION TO DO ALL OF THIS SEPERATELY
img = Image.open('Training_Images/nine_6.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(28, 28))
img = np.array(img)
img = img.astype(float)

kernel = 3
stride = 2

test = CNN(kernel, stride, 0.1)
prediction = test.predict(img)
plot_all(prediction)

images = load_images('Training_Images')