import numpy as np

def add_padding_to_image(img, padding_width):
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,
        img.shape[1] + padding_width * 2
    ))
    img_with_padding[padding_width: -padding_width, padding_width: -padding_width] = img
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