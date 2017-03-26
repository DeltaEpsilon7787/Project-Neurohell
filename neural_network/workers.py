import numpy as np
import numba as nb
from . import utility

__all__ = ["derivate"]

def derivate(args):
    data_in, data_thru, delta, derivatives_amount, original_weights, use_gpu, f, index = args
    delta_array = np.zeros((derivatives_amount))
    delta_array[index] += delta
    mutated_weights = original_weights + delta_array
    mutated_network = utility.merge_weights(data_thru, mutated_weights)
    return (index, (utility.loss_function_at(data_in, mutated_network, use_gpu=use_gpu)-f)/delta)
