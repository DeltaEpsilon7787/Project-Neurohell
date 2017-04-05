import numpy as np

__all__ = ["inject_null_layer", "inject_null_neuron"]

def inject_null_layer(network, inject_after=0):
    if 0 <= inject_after and inject_after < len(network):
        inject_size = network[inject_after].shape[0]
        return (
            network[:inject_after+1]+
            [np.identity(inject_size)]+
            network[inject_after+2:]
            )
    raise ValueError(inject_after)

def inject_null_neuron(network, inject_in=0):
    network_size = len(network)
    if 0 <= inject_in and inject_in < network_size:
        inject_size = network[inject_in].shape[1]
        injected_neuron = np.zeros((1, inject_size))
        if inject_in < network_size-1:
            mutation_size = network[inject_in+1].shape[0]
            mutated_neurons = np.zeros((mutation_size, 1))
            return (
                network[:inject_in] +
                [
                    np.concatenate(
                        (
                            network[inject_in],
                            injected_neuron
                        )
                    )
                ] +
                [
                    np.concatenate(
                        (
                            network[inject_in+1],
                            mutated_neurons,
                        ),
                        axis=1
                    )
                ] +
                network[inject_in+2:]
            )
        return (
            network[:inject_in] +
            [
                np.concatenate(
                    (
                        network[inject_in],
                        injected_neuron
                    )
                )
            ] +
            network[inject_in+1:]
        )
    raise ValueError(inject_in)
