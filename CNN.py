import numpy as np
from plotting_func import plot_image, plot_all
from Base_class import Network
from FCC_layer import FCC

# Convolutional Neural Network structure

class CNN(Network):
    def __init__(self):
        pass
        # self.templates = np.array([
        #     [[1, 1, 1, 1],
        #     [1, -1, -1, 1],
        #     [1, -1, -1, 1],
        #     [1, 1, 1, 1]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0],
        #      [1, 1, 1, 0]],
        #
        #     [[1, 1, 1, -1],
        #      [1, 1, 1, -1],
        #      [-1, -1, 1, -1],
        #      [1, 1, 1, -1]],
        # ])

    # def sigmoid(self, input):
    #     all_images = []
    #     for img in input:
    #         img = img.copy()
    #         for i in range(img.shape[0]):
    #             for j in range(img.shape[0]):
    #                 img[i, j] = 1 / (1 + np.exp(-img[i, j]))
    #         all_images.append(img)
    #     return all_images

    def relu(self, input):
        all_elements = []
        for element in input:
            el = element.copy()
            el[el < 0] = 0
            all_elements.append(el)
        return np.array(all_elements)

    # def relu_deriv(self, input):
    #     all_elements = []
    #     for element in input:
    #         element[element <= 0] = 0
    #         element[element > 0] = 1
    #         all_elements.append(element)
    #     return all_elements

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
                                                 np.array(filter).reshape(self.filters_size ** 2)) + self.bias
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

    def layers(self, input):
        output = self.convolution([input], [self.kernel_a, self.kernel_b, self.kernel_c, self.kernel_d])
        output = self.relu(output)
        output = self.max_pooling(self.return_pools(output))
        output = self.convolution(output, [self.kernel_a, self.kernel_b, self.kernel_c, self.kernel_d])
        output = self.relu(output)
        output = self.max_pooling(self.return_pools(output))

        fcc_1 = FCC(output.shape[0]**2, 100)
        output = fcc_1.forward_prop(output)

        fcc_2 = FCC(output.shape[0]**2, 50)
        output = fcc_2.forward_prop(output)

        fcc_3 = FCC(output.shape[0] ** 2, 10)
        output = fcc_3.forward_prop(output)
        return output

    def predict(self, input):
        output_0 = self.layers(input, 0)
        output_9 = self.layers(input, 9)

        errors = np.array([100, 100 ,100 ,100 ,100 ,100 ,100, 100, 100 ,100])

        output_0 = np.sum(output_0)/len(output_0)
        output_9 = np.sum(output_9) / len(output_9)

        errors[0] = np.abs(np.sum(self.templates[0]) - np.sum(output_0))
        errors[9] = np.abs(np.sum(self.templates[9]) - np.sum(output_9))

        print(f'Error for 0 = {errors[0]}, Error for 9 = {errors[9]}')

        min_value = errors[0]
        index = 0

        for i in range(errors.shape[0]):
            if errors[i] < min_value:
                min_value = errors[i]
                index = i

        return index

    def train(self, dataset, digit):
        iteration = 0
        for image in dataset:
            image = self.layers(image, digit)

            cumulative_error = []

            for img in image:
                img[img == 0] = -1
                # cumulative_sum.append(np.sum(img))
                for i in range(img.shape[0] - 1):
                    for j in range(img.shape[1] - 1):
                        error = self.templates[digit][i, j] - img[i, j]
                        self.weights[digit][i, j] += error * self.learning_rate
                        cumulative_error.append(error)
            iteration += 1
            if iteration % 25 == 0:
                print(f'Error: {np.round(np.abs(np.sum(cumulative_error)/len(cumulative_error)), 3)}')