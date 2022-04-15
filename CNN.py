import numpy as np
from plotting_func import plot_image, plot_all
# Convolutional Neural Network structure

class CNN:
    def __init__(self, filters_size, stride, learning_rate):
        self.outline = np.array([
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1]
        ])

        self.kernel_a = np.array([
            [-1, 2, -1],
            [-1,  2, -1],
            [-1, 2, -1]
        ])

        self.kernel_b = np.array([
            [-1, -1, 2],
            [-1,  2, -1],
            [2, -1, -1]
        ])

        self.kernel_c = np.array([
            [2, -1, -1],
            [-1,  2, -1],
            [-1, -1, 2]
        ])

        self.kernel_d = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])

        self.filters_size = filters_size
        self.stride = stride
        self.weights = np.random.rand(3, 3)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def relu(self, input):
        all_elements = []
        for element in input:
            el = element.copy()
            el[el < 0] = 0
            all_elements.append(el)
        return all_elements

    def relu_deriv(self, input):
        all_elements = []
        for element in input:
            element[element <= 0] = 0
            element[element > 0] = 1
            all_elements.append(element)
        return all_elements

    # def tanh(self, input):
    #     all_elements = []
    #     for element in input:
    #         all_elements.append(np.tanh(element))
    #         print(np.tanh(element))
    #     return np.array(all_elements)
    #
    # def tahn_deriv(self, input):
    #     all_elements = []
    #     for element in input:
    #         new_img = np.empty(shape=element.shape)
    #         for i in range(element.shape[0]):
    #             for j in range(element.shape[1]):
    #                 new_img[i, j] = 1 - np.tanh(element[i, j]) ** 2
    #         all_elements.append(new_img)
    #     return np.array(all_elements)

    def convolution(self, input, filters):
        target_size = input[0].shape[0] - self.filters_size + 1
        container = []

        for image in input:
            for filter in filters:
                filtered_input = np.empty(shape=(target_size, target_size), dtype=int)
                for i in range(target_size):
                    for j in range(target_size):
                        matrix = image[i:i + self.filters_size, j:j + self.filters_size]
                        filtered_input[i, j] = np.dot(np.array(matrix).reshape(self.filters_size ** 2),
                                                 np.array(filter).reshape(self.filters_size ** 2))
                container.append(filtered_input)
        return np.array(container)

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

    def predict(self, input):
        output = self.convolution([input], [self.kernel_a, self.kernel_b, self.kernel_c, self.kernel_d, self.weights])
        output = self.max_pooling(self.return_pools(output))
        output = self.convolution(output, [self.weights])
        output = self.max_pooling(self.return_pools(output))
        output = self.relu(output)
        return output