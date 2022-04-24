import numpy as np
from Base_class import Network
import os

class Convolution(Network):
    def __init__(self, filter_amount):
        self.weights = []

        for x in range(filter_amount):
            self.weights.append(np.random.rand(3, 3) - 0.5)

        self.weights = np.array(self.weights)
        self.bias = np.random.rand() - 0.5

    def forward_prop(self, input):
        self.input = np.array(input)
        weights_size = self.weights[0].shape[0]
        target_size = input[0].shape[1] - weights_size + 1
        container = []
        for image in input:
            for weight in self.weights:
                filtered_input = np.empty(shape=(target_size, target_size), dtype=int)
                for i in range(target_size):
                    for j in range(target_size):
                        matrix = image[i:i + weights_size, j:j + weights_size]
                        # print(weight)
                        filtered_input[i, j] = np.dot(np.array(matrix).reshape(weights_size ** 2),
                                                  np.array(weight).reshape(weights_size ** 2)) + self.bias
                container.append(filtered_input)
        return np.array(container)

    def backward_prop(self, output_error, learning_rate):
        output_error = output_error.reshape(self.input.shape)
        print(output_error)
        os.system('pause')

        cumulative_input_error = 0
        for weight in self.weights:
            input_error = 0
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    input_error += output_error + weight[i, j]
                    weight[i, j] = output_error + weight[i, j]
            cumulative_input_error += input_error / (weight.shape[0]**2)
            self.bias -= learning_rate * output_error
        return cumulative_input_error / len(self.weights)