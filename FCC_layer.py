import numpy as np
from Base_class import Network

class FCC(Network):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_prop(self, input):
        input_data = np.array([])

        for image in input:
            image = image.flatten('C')
            input_data = np.concatenate((input_data, image))

        output = np.dot(self.weights, input_data) + self.bias
        return output

    def backward_prop(self, output_error):
        pass