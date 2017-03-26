import os
import numpy as np
import numba as nb

from multiprocessing import RLock
from random import choice
from PIL import Image

import neural_network

if __name__ == "__main__":
    SIZE = (20, 30)
    def data_from_image(img_path):
        return np.array(
                list(
                        Image.open(img_path)
                        .resize(SIZE)
                        .quantize(2)
                        .tobytes()
                    ),
                dtype=np.float64
                ).transpose()

    data = []

    for directory in os.listdir('training_data')[:]:
        new_data = []
        for file in os.listdir(os.path.join('training_data', directory))[:]:
            new_data.append(
                        data_from_image(
                                os.path.join('training_data',
                                             directory,
                                             file
                                             )
                                )
                    )
        new_data = np.array(new_data)
        data.append(new_data)

    data = np.array(data, dtype=np.float64)

#    some_net = (
#            neural_network.meta_functions.create_layer(5, SIZE[0]*SIZE[1], 1)+
#            neural_network.meta_functions.create_layer(10, 5)+
#            neural_network.meta_functions.create_layer(15, 10)
#            )

    some_net = neural_network.meta_functions.load_network("progress/BEST.neuro")
    net1 = \
    neural_network.training.train_with_random_step(data,
                                                   some_net,
                                                   delta=0.01,
                                                   diminishing_return_cutoff=0.0005,
                                                   lock=RLock())

    neural_network.training.train_with_gradient_descent(data,
                                                        some_net,
                                                        epochs=15,
                                                        utilize_multiprocessing=True,
                                                        lock=RLock(),
                                                        use_n_processes=os.cpu_count())