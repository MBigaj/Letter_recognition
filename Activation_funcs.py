import numpy as np

# SET OF FUNCTIONS FOR CNN NETWORK


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input), axis=0)


def mse(prediction, truth):
    return np.mean(np.power(truth - prediction, 2))


def mse_deriv(prediction, truth):
    return 2 * (prediction - truth) / truth.size


def relu(input):
    all_elements = []
    for element in input:
        el = element.copy()
        el[el < 0] = 0
        all_elements.append(el)
    return np.array(all_elements)


def relu_deriv(input):
    input[input <= 0] = 0
    input[input > 0] = 1
    return input