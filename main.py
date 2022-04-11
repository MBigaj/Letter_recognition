import matplotlib.pyplot
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from convolution_func import *
from pooling_func import *


# IMAGE PLOTTING FUNCTIONS FOR CLARITY

def plot_image(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_two_images(img_1, img_2):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_1, cmap='gray')
    ax[1].imshow(img_2, cmap='gray')
    # ax[0].axis('off')
    # ax[1].axis('off')
    plt.show()

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

kernel_size = 3

# CONVOLUTION LAYER

img = Image.open('Training_Images/pic.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(224, 224))
img = np.array(img)
# plot_image(img)

img_sharpened = convolve(np.array(img), sharpen)
img_blurred = convolve(np.array(img), blur)
img_outlined = convolve(np.array(img), outline)

# print(img_sharpened)

# COMPARISON AFTER FILTERS
# plot_two_images(img, negative_to_zero(img_sharpened))
# plot_two_images(img, img_blurred)
# plot_two_images(img, negative_to_zero(img_outlined))

# PADDING
# img_with_padding_3x3 = add_padding_to_image(np.array(img), get_padding_per_side(kernel_size))
# print(img_with_padding_3x3.shape)
# plot_image(img_with_padding_3x3)


# POOLING LAYER

pool_size = 3 # 3 X 3
stride = 2 # HOW MANY STEPS BETWEEN POOLS

img_pools = get_pools(img, pool_size, stride)
pooled_img = max_pools(img_pools)

img_pools = get_pools(pooled_img, pool_size, stride)
pooled_img = max_pools(img_pools)

img_pools = get_pools(pooled_img, pool_size, stride)
pooled_img = max_pools(img_pools)

img_pools = get_pools(pooled_img, pool_size, stride)
pooled_img = max_pools(img_pools)

plot_two_images(img, pooled_img)