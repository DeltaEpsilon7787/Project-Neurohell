"""
Implements utility functions used everywhere
"""

import numpy as np
import numba as nb
import numba.cuda as cuda

from neural_network import defines
from neural_network import meta_functions

from time import localtime
from itertools import repeat
from collections import deque


def isolate_weights(network):
    weight_array = np.array(())
    for weights in network:
        weight_array = np.append(weight_array, weights.flatten())
    final_array = weight_array.flatten()

    return final_array


def merge_weights(original_network, new_weights):
    network_spread = deque((0,))
    network_shapes = deque()
    for layer in original_network:
        network_spread.append(layer.shape[0]*layer.shape[1])
        network_shapes.append(layer.shape)
    network_spread = np.array(network_spread).cumsum()
    network_spread = zip(network_spread[:-1], network_spread[1:])
    new_network_data = zip(network_spread, network_shapes)
    new_network = deque()
    for data in new_network_data:
        spread, shape = data
        new_network.append(new_weights[spread[0]:spread[1]].reshape(shape).copy())
    return list(new_network)


@nb.jit(nogil=True, cache=True)
def _forward_data_(args):
    data, data_thru, use_gpu = args
    return meta_functions.pass_data(data, data_thru, use_gpu=use_gpu).sum()


@nb.jit(nogil=True, cache=True)
def obtain_interval(data_set, data_thru, use_gpu=False):
    actual_set = list(zip(data_set, repeat(data_thru), repeat(use_gpu)))
    all_hits = list(map(_forward_data_, actual_set))
    return (min(all_hits), max(all_hits))


@nb.jit(nogil=True, cache=True)
def _obtain_interval_set_(data_in, data_thru, use_gpu=False):
    interval_set = deque()
    for data in data_in:
        interval_set.append(obtain_interval(data, data_thru, use_gpu=use_gpu))
    return list(interval_set)


@nb.jit(nogil=True, cache=True)
def loss_function_at(data_in, data_thru, use_gpu=False):
    interval_set = _obtain_interval_set_(data_in,
                                         data_thru,
                                         use_gpu=use_gpu)
    full_loss = defines.loss_function(interval_set)

    return full_loss


def get_hyperparams(data_thru):
    # A tuple of tuples
    # (neurons, inputs)
    data = [layer.shape for layer in data_thru]
    hypers = []
    last_index = 0
    for hyper in data:
        size = last_index + hyper[0]*hyper[1]
        last_index += size
        hypers.append((hyper[0], hyper[1], size))
    return hypers


def timestamp():
    current_time = localtime()
    hour, minute, second = (current_time.tm_hour,
                            current_time.tm_min,
                            current_time.tm_sec)
    current_time_str = "{0:0=2}-{1:0=2}-{2:0=2}".format(hour, minute, second)
    return current_time_str


@cuda.jit
def gpu_matrix_multiplication(column_vector, network_layer, output_layer):
    x, y = cuda.grid(2)

    if (x >= network_layer.shape[0] or
            y >= network_layer.shape[1] or
            y >= column_vector.shape[1] or
            y >= output_layer.shape[1]):
        return

    output_layer[x, y] += column_vector[0, y] * column_vector[x, y]


@nb.jit(nopython=True, nogil=True, cache=True)
def enter_data(data, layer, use_gpu=False):
#    if use_gpu:
#        blockdim = layer.shape
#        griddim = (1, 1)
#        output = np.zeros((1, layer.shape[1]))
#        gpu_matrix_multiplication[griddim, blockdim](data, layer, output)
#        return output
    return defines.sigma(np.dot(layer, data))


@nb.jit(nopython=True, nogil=True, cache=True)
def _intersection_coeff_(interval_a, interval_b):
    best_min = max(interval_a[0], interval_b[0])
    best_max = min(interval_a[1], interval_b[1])

    additional_error = 0
    if abs(best_min-best_max) < defines.CRITICAL_ERROR:
        additional_error = (defines.CRITICAL_ERROR-abs(best_min-best_max))**-5

    least_len = min(interval_a[1]-interval_a[0],
                    interval_b[1]-interval_b[0])

    intersection = ((0, 0)
                    if best_min >= best_max
                    else (best_min, best_max))

    full_len = intersection[1] - intersection[0]
    return (full_len / least_len
            if not (additional_error > 0 or least_len == 0)
            else additional_error)


@nb.jit(nopython=True, nogil=True, cache=True)
def _intersection_cost_(cost):
    return cost**defines.INTERSECTION_COEFF


@nb.jit(nopython=True, nogil=True, cache=True)
def calc_intersect(inter_pair):
    coeff = _intersection_coeff_(inter_pair[0], inter_pair[1])
    return _intersection_cost_(coeff)
