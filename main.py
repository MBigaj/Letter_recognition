import numpy as np
from PIL import Image, ImageOps
import os
from keras.datasets import mnist
from keras.utils import np_utils

from plotting_func import plot_all, plot_image, plot_two_images
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


# LOADING THE TRAINING DATA FROM KERAS

(data, templates), (test_data, test_templates) = mnist.load_data()
data = data.astype(float)
data /= 255
templates = np_utils.to_categorical(templates)

kernel = 3
stride = 2

network = CNN(kernel, stride, 0.1)
network.train(data[0:100], templates[0:100], 25)

img = Image.open('digit.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(28, 28))
img = np.array(img)
img = img.astype(float)
img /= 255

img2 = Image.open('digit_2.jpg')
img2 = ImageOps.grayscale(img2)
img2 = img2.resize(size=(28, 28))
img2= np.array(img2)
img2 = img2.astype(float)
img2 /= 255

img3 = Image.open('digit_3.jpg')
img3 = ImageOps.grayscale(img3)
img3 = img3.resize(size=(28, 28))
img3 = np.array(img3)
img3 = img3.astype(float)
img3 /= 255

# print(network.predict(img))
# print(network.predict(img2))
# print(network.predict(img3))