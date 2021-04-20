from copy import deepcopy
from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from mean_teacher import data


p = Path(__file__).absolute()
PATH = p.parents[1]
DATA_PATH = PATH / 'data'


def cifar10(final_run, val_size=5000):
    num_classes = 10
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023,  0.1994,  0.2010])
    transform_train = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True,
                                transform=transform_train)
    testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True,
                               transform=transform_test)
    if final_run:
        return trainset, testset, testset, num_classes
    valset = deepcopy(testset)
    X_train, X_val, y_train, y_val = train_test_split(trainset.data,
                                                      trainset.targets,
                                                      test_size=val_size,
                                                      stratify=trainset.targets)
    trainset.data, trainset.targets = X_train, y_train
    valset.data, valset.targets = X_val, y_val

    return trainset, valset, testset, num_classes


def cifar100(final_run, val_size=5000):
    num_classes = 100
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675,  0.2565,  0.2761])
    transform_train = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = datasets.CIFAR100(root=DATA_PATH, train=True, download=True,
                                 transform=transform_train)
    testset = datasets.CIFAR100(root=DATA_PATH, train=False, download=True,
                                transform=transform_test)
    if final_run:
        return trainset, testset, testset, num_classes
    valset = deepcopy(testset)
    X_train, X_val, y_train, y_val = train_test_split(trainset.data,
                                                      trainset.targets,
                                                      test_size=val_size,
                                                      stratify=trainset.targets)
    trainset.data, trainset.targets = X_train, y_train
    valset.data, valset.targets = X_val, y_val

    return trainset, valset, testset, num_classes
