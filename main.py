import sys
import numpy
import numpy as np
from PIL import Image, ImageOps

from plotting_func import plot_image, plot_two_images
from convolution_func import tanh, sigmoid, relu, convolve, calc_target_size
from pooling_func import return_pools, max_pooling

# PRESET FILTERS

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125],
    [0.0625, 0.125, 0.0625]
])

outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

weights = np.random.rand(3, 3)

# kernel = filter etc.

# CONVOLUTION LAYER

img = Image.open('Training_Images/nine.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(32, 32))
img = np.array(img)
img = img.astype(float)

# POOLING LAYER

# i = ROWS
# j = COLUMNS
# kernel_size = 3
# pool_size = 3 ... 3x3
# stride = 2, number of steps!!!

kernel_size = 2
pool_size = 2 # 3 X 3
stride = 1 # HOW MANY STEPS BETWEEN POOLS

img = convolve(img, outline, outline.shape[0])
img = relu(img)

pools = return_pools(img, pool_size, stride)
pooled_img = max_pooling(pools)
# plot_image(pooled_img)

conv_2 = convolve(pooled_img, weights, weights.shape[0])
pooled = max_pooling(return_pools(conv_2, pool_size, stride))

# pooled = tanh(pooled)
pooled = relu(pooled)

plot_image(pooled)