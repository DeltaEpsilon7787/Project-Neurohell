from json import dump, load

import numpy as np

from neural_network import utility

__all__ = ["create_layer",
           "pass_data",
           "save_network",
           "load_network",
           "get_mapping_function"]


def create_layer(layer_size, input_size, weight_range=10):
    """
    Creates a random layer of :layer_size
    having :input_size inputs in each neuron
    with random weight from -:weight_range to :weight_range.
    """
    layer = np.random.uniform(-weight_range,
                              +weight_range,
                              (layer_size, input_size))

    return [np.array(layer, dtype=np.float64)]


def pass_data(data_in, data_thru, use_gpu=False):
    """
    Passes given data throught given network possibly using gpu.
    """
    current_data = data_in
    for layer in data_thru:
        try:
            current_data = utility.enter_data(current_data, layer, use_gpu)
        except:
            print(current_data, layer)
    return current_data


def get_mapping_function(marked_data, trained_network):
    data_output = {}
    for marker, data_set in marked_data:
        interval = utility.obtain_interval(data_set, trained_network)
        data_output[interval] = marker

    def mapping(data_in):
        hits = []
        value = pass_data(data_in, trained_network).sum()
        for interval in data_output:
            if interval[0] <= value and value <= interval[1]:
                avg = abs((sum(interval) / 2)-value)
                hits.append((data_output[interval], avg))
        if len(hits):
            return min(hits, key=lambda key: key[1])[0]
        return None

    return mapping

def save_network(network, path='network.neuro', lock=None):
    """
    Saves :network to :path, acquiring :lock if needed
    """
    if lock:
        lock.acquire()
    serializable_network = [layer.tolist() for layer in network]
    with open(path, mode='w') as save_file:
        dump(serializable_network, save_file)
    if lock:
        lock.release()


def load_network(path='network.neuro', lock=None):
    """
    Loads :network from :path, acquiring :lock if needed
    """
    if lock:
        lock.acquire()
    with open(path, mode='r') as load_file:
        unserialized_network = load(load_file)
        return [np.matrix(layer) for layer in unserialized_network]
