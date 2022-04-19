import numpy as np

class Network:
    def __init__(self, filters_size, stride, learning_rate):
        self.kernel_a = np.array([
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1]
        ])

        self.kernel_b = np.array([
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1]
        ])

        self.kernel_c = np.array([
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2]
        ])

        self.kernel_d = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

        self.filters_size = filters_size
        self.stride = stride
        self.learning_rate = learning_rate

    def forward_prop(self, input):
        pass

    def backward_prop(self, output_error):
        pass