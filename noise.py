import random
import pdb

import numpy as np


def generate_noise(dataset_name, dataset, noise_type, noise_ratio):
    num_classes = len(set(dataset.targets))
    noise_matrix = make_noise_matrix(noise_type, noise_ratio, dataset_name, num_classes)
    num_classes = noise_matrix.shape[0]
    num_data = len(dataset.targets)
    num_noise = int(num_data * noise_ratio)
    noisy_idxs = random.sample(range(num_data), k=num_noise)
    noisy_idxs.sort()
    clean_idxs = list(set(range(num_data)) - set(noisy_idxs))
    noisy_gt = []
    for idx in noisy_idxs:
        gt = dataset.targets[idx]
        weights = noise_matrix[gt].tolist()
        del weights[gt]
        noise_list = list(range(num_classes))
        noise_list.remove(gt)
        dataset.targets[idx] = random.choices(noise_list, weights=weights)[0]

    return dataset, clean_idxs, noisy_idxs


def make_noise_matrix(noise_type, noise_ratio, dataset_name, num_classes):
    noise_matrix = np.zeros((num_classes, num_classes))
    if noise_type == 'symmetric':
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    noise_matrix[i][j] = 1.0 - noise_ratio
                else:
                    noise_matrix[i][j] = noise_ratio / (num_classes-1)
    elif noise_type == 'asymmetric':
        if dataset_name == 'cifar10':
            # BIRD  -> AIRPLANE   (2 -> 0)
            # CAT   -> DOG        (3 -> 5)
            # DEER  -> HORSE      (4 -> 7)
            # TRUCK -> AUTOMOBILE (9 -> 1)
            transition = {2: 0,
                          3: 5,
                          4: 7,
                          9: 1}
            for i in range(num_classes):
                if i in transition.keys():
                    noise_matrix[i][i] = 1.0 - noise_ratio
                    noise_matrix[i][transition[i]] = noise_ratio
                else:
                    noise_matrix[i][i] = 1.0

        elif dataset_name == 'cifar100':
            # class i -> (i+1)%100
            for i in range(num_classes):
                noise_matrix[i][i] = 1.0 - noise_ratio
                noise_matrix[i][(i+1)%100] = noise_ratio
    return noise_matrix
