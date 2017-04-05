"""
Implements training.
"""

from time import perf_counter
from itertools import repeat

import multiprocessing as mp
import numpy as np

from neural_network import workers
from neural_network import meta_functions
from neural_network import utility
from neural_network import logging

__all__ = ["train_with_gradient_descent", "train_with_random_step"]


def _get_partial_derivatives_(data_in,
                              data_thru,
                              delta=0.01,
                              log_time=True,
                              utilize_multiprocessing=True,
                              use_gpu=False,
                              pool=None,
                              lock=None):

    REPORT_AFTER = 3
    f_value = utility.loss_function_at(data_in, data_thru)
    original_weights = utility.isolate_weights(data_thru)
    derivatives_amount = len(original_weights)

    def singlethreaded_derivation():
        g_values = []
        logged = False
        start_time = perf_counter()
        mutated_weights = original_weights
        for i in range(derivatives_amount):
            if (log_time and
                    not logged and
                    perf_counter()-REPORT_AFTER >= start_time):

                estimated_time = REPORT_AFTER*derivatives_amount / i
                logging.log("Derivation Time",
                            derivatives_amount,
                            estimated_time,
                            lock=lock)
                logged = True
            mutated_weights[i] += delta
            mutated_network = utility.merge_weights(data_thru, mutated_weights)
            mutated_weights[i] -= delta
            new_loss = utility.loss_function_at(data_in,
                                                mutated_network,
                                                use_gpu=use_gpu)
            g_values.append(new_loss)
        return [(g_value-f_value)/delta for g_value in g_values]

    def multithreaded_derivation():
        data = zip(
            repeat(data_in),
            repeat(data_thru),
            repeat(delta),
            repeat(derivatives_amount),
            repeat(original_weights),
            repeat(use_gpu),
            repeat(f_value),
            range(derivatives_amount)
            )
        if log_time:
            test_data_length = round(derivatives_amount ** 0.5)
            test_data = zip(
                repeat(data_in),
                repeat(data_thru),
                repeat(delta),
                repeat(derivatives_amount),
                repeat(original_weights),
                repeat(use_gpu),
                repeat(f_value),
                range(test_data_length)
                )
            start_time = perf_counter()
            pool.map(workers.derivate,
                     test_data)
            end_time = perf_counter()

            delta_time = end_time - start_time
            multiplier = (1 + derivatives_amount / test_data_length)
            estimated_time = delta_time * multiplier
            logging.log("Derivation Time",
                        derivatives_amount,
                        estimated_time,
                        lock=lock)

        result = pool.map(workers.derivate, data)
        sorted_pairs = sorted(result, key=lambda pair: pair[0])
        final_result = [pair[1] for pair in sorted_pairs]

        return final_result

    if utilize_multiprocessing:
        return multithreaded_derivation()
    return singlethreaded_derivation()


def train_with_gradient_descent(data_in,
                                data_thru,
                                epochs=1,
                                delta=0.01,
                                multiplier=2.0,
                                learning_rate=0.01,
                                satisfactory_error=0.01,
                                diminishing_return_cutoff=0.005,
                                logging_level=logging.ALL,
                                utilize_multiprocessing=False,
                                use_n_processes=4,
                                save_intermediates="progress/Gradient Descent Training.neuro",
                                save_best="progress/BEST.neuro",
                                train_log="logs/train.log",
                                lock=None):

    if logging_level > logging.NONE:
        logging.log("Gradient Training Start",
                    epochs,
                    delta,
                    multiplier,
                    learning_rate,
                    satisfactory_error,
                    diminishing_return_cutoff,
                    save_intermediates,
                    save_best,
                    filename=train_log,
                    lock=lock)

    meta_functions.save_network(data_thru, save_intermediates, lock=lock)

    best_network = data_thru
    best_loss = utility.loss_function_at(data_in, best_network)
    exit_the_loop = False
    pool = mp.Pool(processes=use_n_processes) if utilize_multiprocessing else None

    for epoch in range(epochs):
        current_theta = 0.0  # "Coordinate"
        current_eta = learning_rate  # "Delta coordinate"
        reverse_flag = False
        decelerate_flag = False
        limit = True

        gradient = np.array(_get_partial_derivatives_(data_in,
                                                      best_network,
                                                      delta=delta,
                                                      log_time=logging_level > logging.ADVANCED,
                                                      utilize_multiprocessing=utilize_multiprocessing,
                                                      pool=pool))  # "Units"

        old_weights = np.array(utility.isolate_weights(best_network))  # "Origin"

        past_loss = best_loss

        # There is an imaginary ball rolling/teleporting back and forth
        # Accelerating forward means that this ball rolls against the gradient
        # Decelerating means that it resets its speed to lowest possible

        while True:
            new_weights = -(current_theta+current_eta)*gradient + old_weights
            new_network = utility.merge_weights(best_network, new_weights)
            new_loss = utility.loss_function_at(data_in, new_network)
            if new_loss < best_loss:
                # Continue accelerating
                current_eta *= multiplier
                best_loss = new_loss
                best_network = new_network
                decelerate_flag = False
                reverse_flag = False
                limit = False
                continue
            else:
                # Decelerate or change direction
                if limit:
                    # Special case, came to limit of this network
                    logging.log("Gradient Descent Fail",
                                best_loss,
                                lock=lock)
                    exit_the_loop = True
                    break
                if not decelerate_flag:
                    # Reverse the effects
                    current_theta = current_theta + current_eta / multiplier
                    current_eta = learning_rate
                    decelerate_flag = True
                    continue
                else:
                    # Already decelerated, didn't work, reverse the direction?
                    if not reverse_flag:
                        current_eta = -current_eta
                        reverse_flag = True
                        continue
                    else:
                        # Deceleration didn't work, reversing the direction didn't work, continue on
                        break

        meta_functions.save_network(best_network,
                                    save_intermediates,
                                    lock=lock)

        if best_loss < satisfactory_error:
            if logging_level >= logging.BASIC:
                logging.log("Satisfactory Limit Reached",
                            best_loss,
                            filename=train_log,
                            lock=lock)
            break

        if exit_the_loop:
            break

        delta_loss = 1 - (best_loss/past_loss)
        if delta_loss < diminishing_return_cutoff:
            if logging_level >= logging.BASIC:
                logging.log("Diminishing Return",
                            delta_loss,
                            filename=train_log,
                            lock=lock)
            break

        if logging_level >= logging.ADVANCED:
            logging.log("Epoch Passed",
                        epoch,
                        best_loss,
                        delta_loss,
                        filename=train_log,
                        lock=lock)

    if logging_level >= logging.SUPERFICIAL:
        logging.log("Progress Saved",
                    save_best,
                    filename=train_log,
                    lock=lock)

    meta_functions.save_network(best_network, save_best, lock=lock)
    if pool:
        pool.close()
        pool.join()

    return best_network


def train_with_random_step(data_in,
                           data_thru,
                           delta=0.01,
                           multiplier=2,
                           satisfactory_error=0.01,
                           diminishing_return_cutoff=0.005,
                           guaranteed_epochs=20,
                           logging_level=logging.ALL,
                           bulk_report=0.05,
                           save_intermediates="progress/Random Training.neuro",
                           save_best="progress/BEST.neuro",
                           train_log="logs/train.log",
                           lock=None):
    if logging_level > logging.NONE:
        logging.log("Random Training Start",
                    bulk_report,
                    delta,
                    multiplier,
                    satisfactory_error,
                    diminishing_return_cutoff,
                    guaranteed_epochs,
                    save_intermediates,
                    save_best,
                    filename=train_log,
                    lock=lock)

    meta_functions.save_network(data_thru,
                                save_intermediates,
                                lock=lock)

    best_network = data_thru
    best_loss = utility.loss_function_at(data_in,
                                         best_network)

    old_weights = np.array(utility.isolate_weights(best_network))
    derivatives = len(old_weights)

    current_epoch = 0
    current_theta = 0.0
    current_eta = 1.0
    cum_delta_loss = 0.0
    full_delta = 0.0

    while True:
        past_loss = best_loss
        gradient = np.random.uniform(-delta, delta, (derivatives))
        reverse_flag = False
        decelerate_flag = False
        save_net = False

        while True:
            new_weights = -(current_theta+current_eta)*gradient + old_weights
            new_network = utility.merge_weights(best_network, new_weights)
            new_loss = utility.loss_function_at(data_in, new_network)
            if new_loss < best_loss:
                current_eta *= multiplier
                best_loss = new_loss
                best_network = new_network
                decelerate_flag = False
                reverse_flag = False
                save_net = True
                continue
            else:
                if not decelerate_flag:
                    current_theta = current_theta + current_eta / multiplier
                    current_eta = 1.0
                    decelerate_flag = True
                    continue
                else:
                    if not reverse_flag:
                        current_eta = -current_eta
                        reverse_flag = True
                        continue
                    else:
                        break

        if save_net:
            meta_functions.save_network(data_thru,
                                        save_intermediates,
                                        lock=lock)

        if best_loss < satisfactory_error:
            logging.log("Satisfactory Limit Reached",
                        best_loss,
                        filename=train_log,
                        lock=lock)
            break

        current_epoch += 1
        delta_loss = 1 - (best_loss / past_loss)
        full_delta += delta_loss
        cum_delta_loss += delta_loss
        delta_loss_rate = full_delta / current_epoch
        if current_epoch > guaranteed_epochs:
            if delta_loss_rate < diminishing_return_cutoff:
                logging.log("Diminishing Return",
                            delta_loss_rate,
                            filename=train_log,
                            lock=lock)
                break

        if cum_delta_loss >= bulk_report:
            logging.log("Epoch Passed",
                        current_epoch,
                        best_loss,
                        cum_delta_loss,
                        filename=train_log,
                        lock=lock)
            cum_delta_loss = 0.0
        continue

    logging.log("Progress Saved",
                save_best,
                filename=train_log,
                lock=lock)
    meta_functions.save_network(best_network,
                                save_best,
                                lock=lock)
    return best_network
