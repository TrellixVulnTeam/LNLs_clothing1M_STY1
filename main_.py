import csv
import os
import pdb
from copy import deepcopy
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import datasets
from mean_teacher import data2, losses, optim_weight_swa # data2 : use all labeled samples (according to batchsampler)
from noise import generate_noise
import cli
from train_utils import save_checkpoint, create_model, update_batchnorm, dataloader_filtering
from mt_training import MeanTeacher, Supervised, Filter
import architectures
import pickle


# set working directory to current folder
p = Path(__file__).absolute()
PATH = p.parents[0]
os.chdir(PATH)

args = cli.args
use_cuda = torch.cuda.is_available()
random_number = random.randint(1,100)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

analysis_dict = {'softmax_ema':[],
                 'softmax':[],
                 'clean_labels':[],
                 'noisy_labels':[],
                 'loss_history':[],
                 'labeled_idxs':[],
                 'lr':[]}

"""
################################# Prepare Data ###############################
"""

print('==> Preparing data..')
trainset, valset, testset, num_classes =\
        datasets.__dict__[args.dataset](args.final_run, args.val_size)
args.num_classes = num_classes
analysis_dict['clean_labels'].append(deepcopy(trainset.targets)) # num : 45000
trainset, clean_idxs_train, noisy_idxs_train = generate_noise(args.dataset,
                                                              trainset,
                                                              args.noise_type,
                                                              args.noise_ratio)
analysis_dict['noisy_labels'].append(deepcopy(trainset.targets)) # num : 45000

if args.noisy_validation:
    valset, clean_idxs_val, noisy_idxs_val = generate_noise(args.dataset,
                                                            valset,
                                                            args.noise_type,
                                                            args.noise_ratio)
transform_test = deepcopy(testset.transform)
filtered_trainset = deepcopy(trainset)
trainset.transform = transform_test

filtered_trainloader = DataLoader(filtered_trainset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  pin_memory=True)
trainloader = DataLoader(trainset,
                         batch_size=200,
                         shuffle=False,
                         num_workers=args.workers,
                         pin_memory=True)
valloader = DataLoader(valset,
                       batch_size=200,
                       shuffle=False,
                       num_workers=args.workers,
                       pin_memory=True)
testloader = DataLoader(testset,
                        batch_size=200,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=True)

data_dict = {'filtered_trainloader':filtered_trainloader,
             'trainloader':trainloader,
             'valloader':valloader,
             'testloader':testloader,
             'clean_idxs_train':clean_idxs_train,
             'noisy_idxs_train':noisy_idxs_train,
             'clean_labels':analysis_dict['clean_labels'],
             'noisy_labels':analysis_dict['noisy_labels'],
             'num_classes':num_classes}

"""
########################### Initialize Model ##################################
"""
print('==> Building model..')
model = create_model()
ema_model = create_model(ema = True)
swa_model = create_model(ema = True)
swa_model_optim = optim_weight_swa.WeightSWA(swa_model)
if args.fastswa_frequencies is not None:
    fastswa_freqs = [int(item) for item in args.fastswa_frequencies.split('-')]
    print("Frequent SWAs with frequencies =", fastswa_freqs)
    fastswa_nets = [create_model(ema=True) for _ in fastswa_freqs]
    fastswa_optims = [optim_weight_swa.WeightSWA(fastswa_net) for fastswa_net in fastswa_nets]
    # fastswa_logs = [context.create_train_log("fastswa_validation_freq{}".format(freq)) for freq in fastswa_freqs]

# make directories
name_exp = '_'.join([args.arch, args.dataset, args.noise_type,
                    str(args.noise_ratio), 'lr', str(args.lr),
                    'ema_decay', str(args.ema_decay), 'seed', str(args.seed),
                     args.filtering_type, 'epochs', str(args.first_interval), 'interval', str(args.interval), str(random_number)])

os.makedirs('./results/checkpoint', exist_ok=True)
os.makedirs('./results/log', exist_ok=True)
os.makedirs('./results/plot', exist_ok=True)
os.makedirs('./results/analysis', exist_ok=True)

path_checkpoint = ('./results/checkpoint/' + name_exp)
path_log = ('results/log/' + name_exp + '.csv')
path_plot = ('results/plot/' + name_exp + '.png')
path_analysis = ('results/analysis/' + name_exp + '.pickle')

cudnn.benchmark = True
print('Number of GPUs:', torch.cuda.device_count())
print('Using CUDA..')

"""
################################ Main #########################################
"""
with open(path_log, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc',# 'train clean acc', 'train noisy acc',
                        'valid student loss', 'valid student acc',
                        'valid teacher loss', 'valid teacher acc',
                        'test student acc', 'test teacher acc', 'test swa acc', 'test fastswa acc'])

accuracy = {'train_acc': [],
            'train_clean_acc': [],
            'train_noisy_acc': [],
            'val_acc': [],
            'val_ema_acc': [],
            'test_acc': [],
            'test_ema_acc': [],
            'test_swa_acc': [],
            'test_fastswa_acc': [],
            'precision':[],
            'rate_of_labeled':[]}

# filtering module
f = Filter(trainloader, analysis_dict['clean_labels'], analysis_dict['noisy_labels'], args)

labeled_idxs_history = np.array([], dtype=int) # stack labeled samples' idxs (Full => Inintial labeled idxs(l1) => l1 + l2 => l1 + l2 + l3 => ...)
rate_of_labeled = 100. # start with full labeled dataset
precision = 100. * (1-args.noise_ratio) # precision of full labeled dataset = 1 - noise ratio

optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)
mt = MeanTeacher(model, ema_model, optimizer, args) # mt training
swa = Supervised(swa_model, optimizer, args) # no training
fastswa = Supervised(fastswa_nets[0], optimizer, args) # no training
num_snapshot = 0


# start training
for epoch in range(args.start_epoch, args.first_interval + args.cycles * args.interval +1):

    if epoch < args.first_interval:
        print('In first cycle')
    else:
        cycle = int((epoch - args.first_interval) // args.interval + 2)
        print('In %d -th cycle' %cycle)

    # do the fastSWA updates
    if args.fastswa_frequencies is not None:
        for fastswa_freq, fastswa_net, fastswa_opt in zip(fastswa_freqs, fastswa_nets, fastswa_optims,):
            if epoch >= (args.first_interval - args.interval) and (
                    epoch - args.first_interval + args.interval) % fastswa_freq == 0:
                save_checkpoint(epoch, model, ema_model, swa_model, fastswa_nets[0], accuracy, args, path_checkpoint)
                print("Evaluate fast-swa-{} at epoch {}".format(fastswa_freq, epoch))
                fastswa_opt.update(model)
                update_batchnorm(fastswa_net, trainloader)
                fastswa_acc = fastswa.test(testloader)
                accuracy['test_fastswa_acc'].append(fastswa_acc)
            else:
                accuracy['test_fastswa_acc'].append(None)

    # swa update
    if ((epoch >= args.first_interval)) and ((epoch - args.first_interval) % args.interval) == 0:
        swa_model_optim.update(model)
        print("SWA Model Updated!")
        update_batchnorm(swa_model, trainloader)
        print("Evaluating the SWA model:")
        swa_acc = swa.test(testloader)
        accuracy['test_swa_acc'].append(swa_acc)
    else:
        accuracy['test_swa_acc'].append(None)

    # filtering step
    if args.first_interval is not None and ((epoch >= args.first_interval)) and ((epoch - args.first_interval) % args.interval) == 0: # when new cycle start ( now model is the last model of previous cycle )
        print('Filtering..')
        f.labeled_idxs_history = labeled_idxs_history
        if args.filtering_type == 'snapshot_preds_ensemble':
            num_snapshot += 1
            labeled_idxs = f.filtering(model, args.filtering_type, num_snapshot)
        labeled_idxs_history = np.concatenate((labeled_idxs_history, labeled_idxs)) # update labeled samples' idxs
        unlabeled_idxs = np.setdiff1d(np.arange(len(trainset.targets)), labeled_idxs_history)  # update unlabeled samples' idxs
        rate_of_labeled = 100. * len(labeled_idxs_history) / len(trainset.targets)
        targets = trainloader.dataset.targets
        filtered_trainloader.dataset.targets = \
            [-1 if i in unlabeled_idxs else targets[i] for i in range(len(targets))] # unlabeling
        filtered_trainloader = dataloader_filtering(filtered_trainloader, labeled_idxs_history, unlabeled_idxs, args)

        precision = 100. * len(set(clean_idxs_train) & set(labeled_idxs_history)) / len(labeled_idxs_history)
        recall = 100. * len(set(clean_idxs_train) & set(labeled_idxs_history)) / len(clean_idxs_train)
        print('Filtering Completed! The present dataset\'s Precision: %.3f Recall: %.3f' % (precision, recall))
        print('Labeled: %d\tUnlabeled: %d' % (len(labeled_idxs_history), len(unlabeled_idxs)))

    if epoch == args.first_interval + args.cycles * args.interval: # total last epoch
        test_acc, test_ema_acc = mt.test(testloader)
        swa_acc = swa.test(testloader)
        fastswa_acc = fastswa.test(testloader)
        print('Training Completed!')
        print('Final Accuracy: [Student]Acc: %.3f%%  [Teacher]Acc: %.3f%%  [SWA]Acc: %.3f%%  [fastSWA]Acc: %.3f%%'
              %(test_acc, test_ema_acc, swa_acc, fastswa_acc))
        with open(path_log, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, '_', '_',
                                '_', '_',
                                '_', '_',
                                test_acc, test_ema_acc, swa_acc, fastswa_acc])
        with open(path_analysis, 'wb') as fw:
            pickle.dump(analysis_dict, fw)
        break

    train_loss, train_acc = mt.train(filtered_trainloader, epoch)
    val_loss, val_ema_loss, val_acc, val_ema_acc = mt.validate(valloader)
    test_acc, test_ema_acc = mt.test(testloader)
    accuracy['train_acc'].append(train_acc)
    accuracy['val_acc'].append(val_acc)
    accuracy['val_ema_acc'].append(val_ema_acc)
    accuracy['test_acc'].append(test_acc)
    accuracy['test_ema_acc'].append(test_ema_acc)
    accuracy['precision'].append(precision)
    accuracy['rate_of_labeled'].append(rate_of_labeled)

    # plot
    plt.figure()
    plt.plot(range(epoch+1), accuracy['train_acc'],
             color='blue', label='Train_Acc')
    plt.plot(range(epoch+1), accuracy['val_acc'],
             color='burlywood', label='Valid_Student')
    plt.plot(range(epoch+1), accuracy['val_ema_acc'],
             color='darkorange', label='Valid_Teacher')
    plt.plot(range(epoch+1), accuracy['test_acc'],
             color='limegreen', label='Test_Student')
    plt.plot(range(epoch+1), accuracy['test_ema_acc'],
             color='darkgreen', label='Test_Teacher')
    plt.plot(range(epoch+1), accuracy['test_swa_acc'],
             color='red', label='Test_Swa_Acc', marker='o')
    plt.plot(range(epoch+1), accuracy['test_fastswa_acc'],
             color='firebrick', label='Test_fastSwa_Acc', marker='s')
    plt.plot(range(epoch+1), accuracy['rate_of_labeled'],
             color='lightgrey', label='Rate_of_Labeled', linestyle='--')
    plt.plot(range(epoch+1), accuracy['precision'],
             color='dimgrey', label='Precision', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(path_plot, dpi=300)
    plt.close()

    # log
    with open(path_log, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch+1, train_loss, train_acc,
                            val_loss, val_acc,
                            val_ema_loss, val_ema_acc,
                            test_acc, test_ema_acc, accuracy['test_swa_acc'][-1], accuracy['test_fastswa_acc'][-1]])

analysis_dict['accuracy'] = accuracy
with open(path_analysis, 'wb') as fw:
    pickle.dump(analysis_dict, fw)
