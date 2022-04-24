import numpy as np

class Network:
    def __init__(self):
        raise NotImplementedError

    def forward_prop(self, input):
        raise NotImplementedError

    def backward_prop(self, output_error, learning_rate):
        raise NotImplementedError