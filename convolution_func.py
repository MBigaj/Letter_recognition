import numpy as np


def dtahn(img):
    new_img = np.empty(shape=img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j] = np.round(1 - np.tanh(img[i, j]) ** 2, 3)
    return np.array(new_img)


def tanh(img):
    new_img = np.empty(shape=img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j] = np.round(np.tanh(img[i, j]), 3)
    return np.array(new_img)


def sigmoid(img):
    img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            img[i, j] = 1 / (1 + np.exp(-img[i, j]))
    return img


def relu(img):
    img = img.copy()
    img[img < 0] = 0
    return img


def convolve(img, filter, kernel_size):
    target_size = img.shape[0] - kernel_size + 1
    container = np.empty(shape=(target_size, target_size), dtype=int)

    for i in range(target_size):
        for j in range(target_size):
            matrix = img[i:i + kernel_size, j:j + kernel_size]
            container[i, j] = np.dot(np.array(matrix).reshape(kernel_size ** 2),
                                     np.array(filter).reshape(kernel_size ** 2))
    return container


def calc_target_size(img_size, kernel_size):
    num_pixels = 0

    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1
    return num_pixels