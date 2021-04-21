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

import datasets
from mean_teacher import data, losses
from utils import send_email
from noise import generate_noise
import cli
from train_utils import save_checkpoint, create_model
from warmup_trainer import Supervised, Filter
import architectures
import pickle


from train_utils import lr_fastswa
from utils import progress_bar

# set working directory to current folder
p = Path(__file__).absolute()
PATH = p.parents[0]
os.chdir(PATH)

args = cli.args
use_cuda = torch.cuda.is_available()
CUDA_VISIBLE_DEVICES=0

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# for checking behavior of clean and noisy samples
analysis_dict = {'softmax':[],
                 'softmax_ema':[],
                 'clean_labels':[],
                 'noisy_labels':[],
                 'loss_history':[],
                 'lr':[]}

"""
################################# Prepare Data ###############################
"""

print('==> Preparing data..')
trainset, valset, testset, num_classes =\
        datasets.__dict__[args.dataset](args.final_run, args.val_size)
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

# for saving dataset
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
model_factory = architectures.__dict__[args.arch]
model = model_factory(args.pretrained, num_classes)
model = nn.DataParallel(model).cuda()

name_exp = '_'.join([args.arch, args.dataset, args.noise_type,
                    str(args.noise_ratio), 'lr', str(args.lr), 'lr_min', str(args.lr_min),
                    'seed', str(args.seed), 'interval', str(args.interval), 'cycles', str(args.cycles)])

# make directories
os.makedirs('./results/warmup/checkpoint', exist_ok=True)
os.makedirs('results/warmup/log', exist_ok=True)
os.makedirs('results/warmup/plot', exist_ok=True)
os.makedirs('results/warmup/analysis', exist_ok=True)

path_log = ('results/warmup/log/' + name_exp + '.csv')
path_plot = ('results/warmup/plot/' + name_exp + '.png')
path_plot_lr = ('results/warmup/plot/' + name_exp + '_lr.png')
path_analysis = ('results/warmup/analysis/' + name_exp + '.pickle')
path_dataset = ('results/warmup/data_dict.pickle')
with open(path_dataset, 'wb') as fw:
    pickle.dump(data_dict, fw)

cudnn.benchmark = True
print('Number of GPUs:', torch.cuda.device_count())
print('Using CUDA..')

class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
if args.consistency_type == 'mse':
    consistency_criterion = losses.softmax_mse_loss
elif args.consistency_type == 'kl':
    consistency_criterion = losses.softmax_kl_loss
else:
    assert False, args.consistency_type
residual_logit_criterion = losses.symmetric_mse_loss

"""
################################ Main #########################################
"""

with open(path_log, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc', 'clean acc', 'noisy acc', 'noisy memorized',
                        'valid loss', 'valid acc',
                        'test acc'])

# filtering or finding behavior value module
f = Filter(trainloader, analysis_dict['clean_labels'], analysis_dict['noisy_labels'], args)
accuracy = {'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'clean_acc': [],
            'noisy_acc': [],
            'noisy_memorized':[]}

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)

# supervised learning module
sl = Supervised(model, optimizer, args)
epoch = 0
while epoch < args.interval * args.cycles: # for warm-up analysis with full labeled dataset

    # check lr for every epoch
    analysis_dict['lr'].append(sl.optimizer.param_groups[0]['lr'])
    train_loss, train_acc = sl.train(filtered_trainloader, epoch)
    val_loss, val_acc = sl.validate(valloader)
    test_acc = sl.test(testloader)
    accuracy['train_acc'].append(train_acc)
    accuracy['val_acc'].append(val_acc)
    accuracy['test_acc'].append(test_acc)

    softmax, softmax_ema, loss, clean_acc, noisy_acc, noisy_memorized = f.get_behavior(model)
    accuracy['clean_acc'].append(clean_acc)
    accuracy['noisy_acc'].append(noisy_acc)
    accuracy['noisy_memorized'].append(noisy_memorized)
    analysis_dict['softmax'].append(deepcopy(softmax))
    analysis_dict['softmax_ema'].append(deepcopy(softmax_ema))
    analysis_dict['loss_history'].append(deepcopy(loss))

    epoch += 1
    # plot
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(range(epoch), accuracy['train_acc'],
             color='blue', label='Train_Acc')
    ax1.plot(range(epoch), accuracy['val_acc'],
             color='orange', label='Valid_Acc')
    ax1.plot(range(epoch), accuracy['test_acc'],
             color='green', label='Test_Acc')
    ax1.plot(range(epoch), accuracy['clean_acc'],
             color='midnightblue', label='Clean_Acc')
    ax1.plot(range(epoch), accuracy['noisy_acc'],
             color='cornflowerblue', label='Noisy_Acc')
    ax1.plot(range(epoch), accuracy['noisy_memorized'],
             color='darkred', label='Noisy_Memorized')
    ax1.legend(loc='lower right')
    ax1.grid(True)


    ax2 = ax1.twinx()
    ax2.set_ylabel('LR')
    ax2.plot(range(epoch), analysis_dict['lr'],
             color='dimgray', label='LR', linestyle='--')
    ax2.legend(loc='upper right')

    fig.savefig(path_plot, dpi=300)
    plt.close()

    # log
    with open(path_log, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc,
                            clean_acc, noisy_acc, noisy_memorized,
                            val_loss, val_acc,
                            test_acc])


analysis_dict['accuracy'] = accuracy
with open(path_analysis, 'wb') as fw:
    pickle.dump(analysis_dict, fw)