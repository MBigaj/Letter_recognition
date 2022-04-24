import numpy as np
from plotting_func import plot_image, plot_all
from Base_class import Network
from FCC_layer import FCC
from Convolution_layer import Convolution
import os

from Activation_funcs import relu, relu_deriv, mse, mse_deriv, softmax

# Convolutional Neural Network structure

class CNN(Network):
    def __init__(self, filters_size, stride, learning_rate):
        self.filters_size = filters_size
        self.stride = stride
        self.learning_rate = learning_rate
        self.bias = np.random.rand() - 0.5
        self.conv_1 = Convolution(4)
        self.conv_2 = Convolution(4)
        self.fcc_1 = FCC(256, 100)
        self.fcc_2 = FCC(100, 50)
        self.fcc_3 = FCC(50, 10)

    def flatten(self, input):
        output = np.array([])

        for image in input:
            image = image.flatten('C')
            output = np.concatenate((output, image))
        output = output.reshape(1, output.shape[0])

        return output

    # def sigmoid(self, input):
    #     all_images = []
    #     for img in input:
    #         img = img.copy()
    #         for i in range(img.shape[0]):
    #             for j in range(img.shape[0]):
    #                 img[i, j] = 1 / (1 + np.exp(-img[i, j]))
    #         all_images.append(img)
    #     return all_images

    # def convolution(self, input, filters):
    #     target_size = input[0].shape[0] - self.filters_size + 1
    #     container = []
    #
    #     for image in input:
    #         for filter in filters:
    #             filtered_input = np.empty(shape=(target_size, target_size), dtype=int)
    #             for i in range(target_size):
    #                 for j in range(target_size):
    #                     matrix = image[i:i + self.filters_size, j:j + self.filters_size]
    #                     filtered_input[i, j] = np.dot(np.array(matrix).reshape(self.filters_size ** 2),
    #                                              np.array(filter).reshape(self.filters_size ** 2)) + self.bias
    #             container.append(filtered_input)
    #     return np.array(container)

    def return_pools(self, input):
        all_pools = []
        for filtered_input in input:
            input_pool = []
            for i in np.arange(filtered_input.shape[0], step=self.stride):
                for j in np.arange(filtered_input.shape[0], step=self.stride):
                    single_pool = filtered_input[i:i + self.filters_size, j:j + self.filters_size]
                    if single_pool.shape == (self.filters_size, self.filters_size):
                        input_pool.append(single_pool)
            all_pools.append(input_pool)
        return np.array(all_pools)

    def max_pooling(self, input):
        target_shape = (int(np.sqrt(input[0].shape[0])), int(np.sqrt(input[0].shape[0])))
        new_outputs = []
        for pools in input:
            max_pools = []
            for pool in pools:
                max_pools.append(np.max(pool))
            max_pools = np.array(max_pools).reshape(target_shape)
            new_outputs.append(max_pools)
        return np.array(new_outputs)

        # APPLY RANDOM WEIGHTS TO CONVOLUTION and add BACK PROP to convolution
    def layers(self, input):
        output = self.conv_1.forward_prop([input])
        output = relu(output)
        output = self.max_pooling(self.return_pools(output))
        output = self.conv_2.forward_prop(output)
        output = relu(output)
        output = self.max_pooling(self.return_pools(output))
        output = self.flatten(output)
        output = self.fcc_1.forward_prop(output)
        output = self.fcc_2.forward_prop(output)
        output = self.fcc_3.forward_prop(output)
        output = softmax(output[0])
        return output

    def predict(self, input):
        output = self.layers(input)
        return output

    def train(self, dataset, templates, epochs):
        data_size = len(dataset)

        for i in range(epochs):
            err = 0
            for j in range(data_size):
                output = dataset[j]
                output = self.layers(output)

                err += mse(output, templates[j])

                error = mse_deriv(output, templates[j])
                error = np.array(error).reshape((1, 10))
                error = self.fcc_3.backward_prop(error, self.learning_rate)
                error = self.fcc_2.backward_prop(error, self.learning_rate)
                error = self.fcc_1.backward_prop(error, self.learning_rate)
                error = relu_deriv(error)
                # error = self.conv_2.backward_prop(error, self.learning_rate)
                # error = self.conv_1.backward_prop(error, self.learning_rate)
            err /= data_size
            print(f'Epoch: {i+1}/{epochs} - Error: {err}')