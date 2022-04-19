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

img = Image.open('Training_Images/nine/nine_1.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(28, 28))
img = np.array(img)
img = img.astype(float)

kernel = 3
stride = 2

network = CNN(kernel, stride, 0.5)

images_of_nine = load_images('Training_Images/nine')
images_of_zero = load_images('Training_images/zero')

images_of_nine = [el / 255.0 for el in images_of_nine]
images_of_zero = [el / 255.0 for el in images_of_zero]


# network.train(images_of_zero, 0)
# print()
# network.train(images_of_nine, 9)

img /= 255.0

# prediction = network.layers(img, 0)

# img = img.flatten('C')
#
# weights = np.random.rand(28**2, 100) - 0.5
# bias = np.random.rand(1, 100) - 0.5
# output = np.dot(img, weights) + bias
# output = np.array(output)
# print(output.shape)

network.layers(img)