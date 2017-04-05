"""
Genetic algorithm works as follows:
1. Start from a single neuron network.
2. Train it.
3. Evaluate its worthness
4. To each network in pool add a null-neuron to each layer and add null-layer
5. Train all networks
6. Leave only Q% of best networks
7. Repeat from 4 until satisfactory limit has been reached
8. End
"""

import multiprocessing as mp
from math import ceil
from genetic_evolution import evolution
from neural_network import logging, meta_functions, utility, training

def _inject_id(params, id_counter):
    new_params = dict(params)
    new_params["save_intermediates"] = (
            params["save_intermediate_prefix"] +
            str(id_counter) +
            ".neuro")
    new_params["save_best"] = (
            params["save_best_prefix"] +
            str(id_counter) +
            ".neuro")
    new_params["train_log"] = (
            params["train_log_prefix"] +
            str(id_counter) +
            ".log"
            )
    return new_params


def _prune_params(params):
    new_params = dict(params)
    del new_params['save_intermediate_prefix']
    del new_params['save_best_prefix']
    del new_params['train_log_prefix']
    return new_params


def _get_random_training_params():
    param_file = open('./config/random_training.params')
    params = dict((("delta", 0.01),
                   ("multiplier", 2.0),
                   ("satisfactory_error", 0.01),
                   ("diminishing_return_cutoff", 0.005),
                   ("guaranteed_epochs", 20),
                   ("bulk_report", 0.05),
                   ("save_intermediate_prefix", "progress/life/intermediate/random_train_"),
                   ("save_best_prefix", "progress/life/best/random_train_"),
                   ("train_log_prefix", "logs/life/random_train_")))

    param_converters = dict((("delta", float),
                             ("multiplier", float),
                             ("satisfactory_error", float),
                             ("diminishing_return_cutoff", float),
                             ("guaranteed_epochs", int),
                             ("bulk_report", float),
                             ("save_intermediate_prefix", str),
                             ("save_best_prefix", str),
                             ("train_log_prefix", str)))

    for line in param_file.readlines():
        param, value = line.split("=")
        value = value.split("\n")[0]
        if param not in params:
            param_file.close()
            raise ValueError("Unknown param in random_training.params :: "+param)
        params[param] = param_converters[param](value)

    param_file.close()
    return params


def _get_gradient_training_params():
    param_file = open('./config/gradient_training.params')
    params = dict((("epochs", 1),
                   ("delta", 0.01),
                   ("multiplier", 2.0),
                   ("learning_rate", 0.01),
                   ("satisfactory_error", 0.005),
                   ("diminishing_return_cutoff", 20),
                   ("save_intermediate_prefix", "progress/life/intermediate/gradient_train_"),
                   ("save_best_prefix", "progress/life/best/gradient_train_"),
                   ("train_log_prefix", "logs/life/gradient_train_")))
    param_converters = dict((("epochs", int),
                             ("delta", float),
                             ("multiplier", float),
                             ("learning_rate", float),
                             ("satisfactory_error", float),
                             ("diminishing_return_cutoff", float),
                             ("save_intermediate_prefix", str),
                             ("save_best_prefix", str),
                             ("train_log_prefix", str)))

    for line in param_file.readlines():
        param, value = line.split("=")
        value = value.split("\n")[0]
        if param not in params:
            param_file.close()
            raise ValueError("Unknown param in gradient_training.params :: "+param)
        params[param] = param_converters[param](value)

    param_file.close()
    return params


def life(training_data_set,
         satisfactory_limit=1,
         weight_range=10000,
         size=(40, 60),
         logging_level=logging.ALL,
         coeff_q=0.2,
         multitraining=False,
         use_n_processes=2):

    network_pool = [meta_functions.create_layer(1,
                                                size[0]*size[1],
                                                weight_range)]
#    process_pool = (
#            mp.Pool(processes=use_n_processes)
#            if multitraining
#            else None
#            )

    id_counter = 1
    generation_counter = 1

    random_training_params = _get_random_training_params()
    gradient_training_params = _get_gradient_training_params()

    def value_function(data_thru):
        return utility.loss_function_at(training_data_set, data_thru)

    best_loss = value_function(network_pool[0])

    while best_loss > satisfactory_limit:
        trained_networks = []
        for network in network_pool:
            current_params = _inject_id(random_training_params, id_counter)
            current_params = _prune_params(current_params)
            trained_network = training.train_with_random_step(training_data_set,
                                                              network,
                                                              **current_params,
                                                              logging_level=logging_level)
            id_counter += 1
            trained_networks.append(trained_network)

        network_pool = []
        for network in trained_networks:
            current_params = _inject_id(gradient_training_params, id_counter)
            current_params = _prune_params(current_params)
            trained_network = training.train_with_gradient_descent(training_data_set,
                                                                   network,
                                                                   **current_params,
                                                                   logging_level=logging_level)
            id_counter += 1
            network_pool.append(trained_network)

        network_values = map(value_function, network_pool)
        final_networks = zip(network_pool, network_values)
        worthy_networks = sorted(final_networks, key=lambda m: m[1])
        new_best_loss = worthy_networks[0][1]
        leave_only_this = ceil(len(worthy_networks) * coeff_q)
        best_networks = worthy_networks[:leave_only_this]
        network_pool = []
        for best_network in best_networks:
            for inject_at in range(len(best_network[0])):
                network_pool.append(evolution.inject_null_layer(best_network[0],
                                                                inject_at))
                network_pool.append(evolution.inject_null_neuron(best_network[0],
                                                                 inject_at))
        if logging_level >= logging.BASIC:
            logging.log("Life Log",
                        best_loss,
                        new_best_loss,
                        len(network_pool),
                        generation_counter,
                        filename='life.log')

        best_loss = new_best_loss
        generation_counter += 1