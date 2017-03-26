import numpy as np


class VirtualNetwork:
    data = None
    layers = None

    def __init__(self, data, source):
        self.data = data
        for layer in source:
            height, width = layer.shape
            new_layer = Layer(height, width)
            for neuron in layer:
                new_layer.insert_neuron(neuron)
            self.layers.append(new_layer)

    def calculate(self):
        current_data = self.data
        for layer in self.layers:
            current_data = layer.calculate(current_data)
        return current_data

    def mutate(self, input_id, delta):
        current_id = 0

        layer_id = 0
        local_neuron_id = -1
        local_weight_id = -1

        for layer in self.layers:
            if current_id < layer.layer_size:
                layer_id += 1
                current_id += layer.layer_size
            else:
                local_neuron_id = (input_id-current_id) // layer.input_size
                local_weight_id = (input_id-current_id) % layer.input_size
                break
        if local_neuron_id == -1:
            raise ValueError("This ID is not part of the network")

        self.layers[layer_id].mutate(local_neuron_id, local_weight_id, delta)
        for layer in self.layers[layer_id+1:]:
            layer.reset()


class Layer:
    weight_arrays = None
    memoized_values = None
    invariant_table = None
    input_size = 0
    layer_size = 0

    def __init__(self, layer_size, input_size):
        self.input_size = input_size
        self.layer_size = layer_size

    def calculate(self, data):
        data = data.transpose()
        output = []
        for index in range(self.layer_size):
            weights = self.weight_arrays[index]
            invariant = self.invariant_table[index]
            memoized_value = self.memoized_values[index]
            if invariant:
                output.append(memoized_value)
            else:
                value = (weights*data).sum()
                self.invariant_table[index] = True
                self.memoized_values[index] = value
                output.append(value)

        return np.array(output).transpose()

    def mutate(self, neuron_id, weight_id, delta):
        self.invariant_table[neuron_id] = False
        self.weight_arrays[neuron_id][weight_id] += delta

    def reset(self):
        self.invariant_table = [False for i in range(self.layer_size)]

