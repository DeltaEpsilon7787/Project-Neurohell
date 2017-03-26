"""
Some legacy code that still works, but was superseeded by
  JIT compilation and removed because it is not supported.

Most imports are inside functions/classes themselves.
"""


def loss_function(interval_set: 'List of intervals'):
    """
    The function to be minimized.

    defines.py
    """
    from itertools import combinations
    from neural_network.utility import calc_intersect

    all_combinations = combinations(interval_set, 2)
    intersections = map(calc_intersect, all_combinations)
    return sum(intersections)


def mutate_network(data_thru, neuron_id, delta):
    """
    Irretrivably mutates given network at neuron :neuron_id adding :delta

    meta_functions.py
    """
    from neural_network.utility import get_hyperparams

    hyper_params = get_hyperparams(data_thru)

    layer_indx = 0
    neuron_id = 0
    for params in hyper_params:
        if neuron_id < params[2]:  # Index
            size = params[0]*params[1]
            local_index = neuron_id - params[2] - size
            row = local_index // params[1]
            col = local_index % params[1]
            data_thru[layer_indx][row, col] += delta
            break
        else:
            neuron_id += params[0]
