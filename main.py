import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def add_padding_to_image(img, padding_width):
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,
        img.shape[1] + padding_width * 2
    ))
    img_with_padding[-padding_width, -padding_width] = img
    return img_with_padding


def get_padding_per_side(kernel_size):
    return kernel_size // 2


def negative_to_zero(img):
    img = img.copy()
    img[img < 0] = 0
    return img


def convolve(img, kernel):
    k = kernel.shape[0]
    target_size = calc_target_size(img.shape[0], k)
    convolved_img = np.zeros(shape=(target_size, target_size))

    for i in range(target_size):
        for j in range(target_size):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
    return convolved_img


def calc_target_size(img_size, kernel_size):
    num_pixels = 0

    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1
    return num_pixels


def plot_image(img: np.array):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def plot_two_images(img_1, img_2):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_1, cmap='gray')
    ax[1].imshow(img_2, cmap='gray')
    plt.show()


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


img = Image.open('Training_Images/letter_A.jpg')
img = ImageOps.grayscale(img)
img = img.resize(size=(224, 224))
# plot_image(img)

img_sharpened = convolve(np.array(img), sharpen)
img_blurred = convolve(np.array(img), blur)
img_outlined = convolve(np.array(img), outline)

# print(img_sharpened)

# plot_two_images(img, negative_to_zero(img_sharpened))
plot_two_images(img, negative_to_zero(img_outlined))