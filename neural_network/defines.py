"""
Includes functions that define a neural network:
Functions:
    loss_function :: to be minimized
    sigma :: applied after multiplication in one layer
"""

from numpy import exp
from numba import jit
import numba as nb

from neural_network import utility

INTERSECTION_COEFF = 2
CRITICAL_ERROR = 0.1


@jit(nopython=True, nogil=True, cache=True)
def loss_function(interval_set):
    """
    The function to be minimized.
    """
    loss_sum = 0
    for i in range(len(interval_set)-1):
        for j in range(i+1, len(interval_set)):
            inter_pair = (interval_set[i], interval_set[j])
            loss_sum += utility.calc_intersect(inter_pair)
    return loss_sum


@jit(nopython=True, nogil=True, cache=True)
def sigma(val):
    """
    Sigmoid function to be applied.
    """
    # return 1 / (1+exp(-val))
    return val
