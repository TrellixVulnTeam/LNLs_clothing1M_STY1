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
from train_utils import save_checkpoint
from mt_standard import Supervised, Filter
import architectures
import pickle

# from train_utils import adjust_learning_rate_fastswa
from utils import progress_bar

# set working directory to current folder
p = Path(__file__).absolute()
PATH = p.parents[0]
os.chdir(PATH)

args = cli.args
use_cuda = torch.cuda.is_available()
CUDA_VISIBLE_DEVICES=2

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

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
# for i in range(10):

print('==> Building model..')
model_factory = architectures.__dict__[args.arch]
model = model_factory(args.pretrained, num_classes)
model = nn.DataParallel(model).cuda()
# state = {'model_weight':model.state_dict()}
# path_checkpoint1 = ('./results/cyclic/checkpoint/model')
# checkpoint = torch.load(path_checkpoint1)
# model.load_state_dict(checkpoint['model_weight'])

name_exp = '_'.join([args.arch, args.dataset, args.noise_type,
                    str(args.noise_ratio), 'lr', str(args.lr),
                    'ema_decay', str(args.ema_decay), 'seed', str(args.seed), 'tmax', str(args.interval)])

# make directories
os.makedirs('./results/cyclic/checkpoint', exist_ok=True)
os.makedirs('results/cyclic/log', exist_ok=True)
os.makedirs('results/cyclic/plot', exist_ok=True)

# path_checkpoint = ('./results/cyclic/checkpoint/' + name_exp)
# path_checkpoint1 = ('./results/cyclic/checkpoint/model')
# torch.save(model, path_checkpoint1)
# pdb.set_trace()
path_log = ('results/cyclic/log/' + name_exp + '.csv')
path_plot = ('results/cyclic/plot/' + name_exp + '.png')
path_plot_lr = ('results/cyclic/plot/' + name_exp + '_lr.png')
path_analysis = ('results/cyclic/analysis/' + name_exp + '.pickle')
path_data = ('results/cyclic/data_dict.pickle')
with open(path_data, 'wb') as fw:
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

f = Filter(trainloader, analysis_dict['clean_labels'], analysis_dict['noisy_labels'], args)
iter_filtering = 0
accuracy = {'train_acc': [],
            'val_acc': [],
            'test_acc': [],
            'clean_acc': [],
            'noisy_acc': [],
            'noisy_memorized':[]}
epoch = 0
stop_train = False

# num_wait = 0
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)
sl = Supervised(model, optimizer, args)
# lr_list = list()
while epoch < 300: #args.max_total_epochs: # 600

    # lr_list.append(sl.optimizer.param_groups[0]['lr'])
    analysis_dict['lr'].append(sl.optimizer.param_groups[0]['lr'])
    train_loss, train_acc = sl.train(filtered_trainloader, epoch)
    val_loss, val_acc = sl.validate(valloader, epoch)
    test_acc = sl.test(testloader)
    accuracy['train_acc'].append(train_acc)
    accuracy['val_acc'].append(val_acc)
    accuracy['test_acc'].append(test_acc)


    softmax, softmax_ema, loss, clean_acc, noisy_acc, noisy_memorized = f.filter(model)
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

    # plt.figure()
    # plt.plot(range(epoch), accuracy['train_acc'],
    #          color='blue', label='Train_Acc')
    # plt.plot(range(epoch), accuracy['val_acc'],
    #          color='orange', label='Valid_Acc')
    # plt.plot(range(epoch), accuracy['test_acc'],
    #          color='green', label='Test_Acc')
    # plt.plot(range(epoch), accuracy['clean_Acc'],
    #          color='midnightblue', label='Clean_Acc')
    # plt.plot(range(epoch), accuracy['noisy_Acc'],
    #          color='cornflowerblue', label='Noisy_Acc')
    # plt.plot(range(epoch), accuracy['noisy_memorized'],
    #          color='darkred', label='Noisy_Memorized')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True, axis='y')
    # plt.savefig(path_plot, dpi=300)
    # plt.close()
    #
    # plt.figure()
    # plt.plot(range(epoch), analysis_dict['lr'],
    #          color='blue', label='lr')
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning rate')
    # plt.legend()
    # plt.grid(True, axis='y')
    # plt.savefig(path_plot_lr, dpi=300)
    # plt.close()

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

# analysis_dict = {'softmax':[],
#                  'clean_labels':[],
#                  'noisy_labels':[],
#                  'loss_history':[]}
#
# """
# ################################# Prepare Data ###############################
# """
#
#
# print('==> Preparing data..')
# trainset, valset, testset, num_classes =\
#         datasets.__dict__[args.dataset](args.final_run, args.val_size)
# analysis_dict['clean_labels'].append(deepcopy(trainset.targets)) # num : 45000
#
# trainset, clean_idxs_train, noisy_idxs_train = generate_noise(args.dataset,
#                                                               trainset,
#                                                               args.noise_type,
#                                                               args.noise_ratio)
# analysis_dict['noisy_labels'].append(deepcopy(trainset.targets)) # num : 45000
#
# if args.noisy_validation:
#     valset, clean_idxs_val, noisy_idxs_val = generate_noise(args.dataset,
#                                                             valset,
#                                                             args.noise_type,
#                                                             args.noise_ratio)
# transform_test = deepcopy(testset.transform)
# filtered_trainset = deepcopy(trainset)
# trainset.transform = transform_test
#
# filtered_trainloader = DataLoader(filtered_trainset,
#                                   batch_size=args.batch_size,
#                                   shuffle=True,
#                                   num_workers=args.workers,
#                                   pin_memory=True)
# trainloader = DataLoader(trainset,
#                          batch_size=200,
#                          shuffle=False,
#                          num_workers=args.workers,
#                          pin_memory=True)
# valloader = DataLoader(valset,
#                        batch_size=200,
#                        shuffle=False,
#                        num_workers=args.workers,
#                        pin_memory=True)
# testloader = DataLoader(testset,
#                         batch_size=200,
#                         shuffle=False,
#                         num_workers=args.workers,
#                         pin_memory=True)
#
# data_dict = {'filtered_trainloader':filtered_trainloader,
#              'trainloader':trainloader,
#              'valloader':valloader,
#              'testloader':testloader,
#              'clean_idxs_train':clean_idxs_train,
#              'noisy_idxs_train':noisy_idxs_train,
#              'num_classes':num_classes}
#
# """
# ########################### Initialize Model ##################################
# """
#
#
# print('==> Building model..')
# model_factory = architectures.__dict__[args.arch]
# model = model_factory(args.pretrained, num_classes)
# model = nn.DataParallel(model).cuda()
# pdb.set_trace()
# name_exp = '_'.join([args.arch, args.dataset, args.noise_type,
#                     str(args.noise_ratio), 'lr', str(args.lr),
#                     'ema_decay', str(args.ema_decay), 'seed', str(args.seed)])
#
# # make directories
# os.makedirs('./results/standard/checkpoint', exist_ok=True)
# os.makedirs('results/standard/log', exist_ok=True)
# os.makedirs('results/standard/plot', exist_ok=True)
#
# path_checkpoint = ('./results/standard/checkpoint/' + name_exp)
# path_log = ('results/standard/log/' + name_exp + '.csv')
# path_plot = ('results/standard/plot/' + name_exp + '.png')
#
# path_data = ('results/data_dict.pickle')
# with open(path_data, 'wb') as fw:
#     pickle.dump(data_dict, fw)
# pdb.set_trace()
#
# cudnn.benchmark = True
# print('Number of GPUs:', torch.cuda.device_count())
# print('Using CUDA..')
#
# class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
# if args.consistency_type == 'mse':
#     consistency_criterion = losses.softmax_mse_loss
# elif args.consistency_type == 'kl':
#     consistency_criterion = losses.softmax_kl_loss
# else:
#     assert False, args.consistency_type
# residual_logit_criterion = losses.symmetric_mse_loss
#
# """
# ################################ Main #########################################
# """
#
# with open(path_log, 'w') as logfile:
#     logwriter = csv.writer(logfile, delimiter=',')
#     logwriter.writerow(['epoch', 'train loss', 'train acc',
#                         'valid student loss', 'valid student acc',
#                         'valid teacher loss', 'valid teacher acc',
#                         'test student acc', 'test teacher acc'])
#
# f = Filter(trainloader, args)
# iter_filtering = 0
# accuracy = {'train_acc': [],
#             'val_acc': [],
#             'val_ema_acc': [],
#             'test_acc': [],
#             'test_ema_acc': []}
# epoch = 0
# stop_train = False
#
# while epoch < 150: #args.max_total_epochs: # 600
#     optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                           momentum=args.momentum,
#                           weight_decay=args.decay, nesterov=args.nesterov)
#     sl = Supervised(model, optimizer, args)
#     train_loss, train_acc = sl.train(filtered_trainloader, epoch)
#     val_loss, val_acc = sl.validate(valloader, epoch)
#     test_acc = sl.test(testloader)
#     accuracy['train_acc'].append(train_acc)
#     accuracy['val_acc'].append(val_acc)
#     accuracy['test_acc'].append(test_acc)
#
#     epoch += 1
#     # plot
#     plt.figure()
#     plt.plot(range(epoch), accuracy['train_acc'],
#              color='blue', label='Train_Student')
#     plt.plot(range(epoch), accuracy['val_acc'],
#              color='orange', label='Valid_Student')
#     plt.plot(range(epoch), accuracy['test_acc'],
#              color='green', label='Test_Student')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True, axis='y')
#     plt.savefig(path_plot, dpi=300)
#     plt.close()
#     # log
#     with open(path_log, 'a') as logfile:
#         logwriter = csv.writer(logfile, delimiter=',')
#         logwriter.writerow([epoch, train_loss, train_acc,
#                             val_loss, val_acc,
#                             test_acc])
#     if val_acc == max(accuracy['val_acc']):
#         filtering_model = deepcopy(sl.model)
#         print('\033[33m' + '*****************Saved*****************' + '\033[37m')
#         state = {'model_weight':sl.model.state_dict()}
#         torch.save(state, path_checkpoint)
#
#     labeled_idxs, unlabeled_idxs, softmax, softmax_ema, loss = f.filter(filtering_model)
#     analysis_dict['softmax'].append(deepcopy(softmax))
#     analysis_dict['softmax_ema'].append(deepcopy(softmax_ema))
#     analysis_dict['loss_history'].append(deepcopy(loss))
#
#     # # Filtering step
#     # print('Filtering..')
#     # checkpoint = torch.load(path_checkpoint)
#     # filtering_model = model_factory(args.pretrained, num_classes)
#     # filtering_model = nn.DataParallel(filtering_model).cuda()
#     # filtering_model.load_state_dict(checkpoint['ema_model_weight']) # upload the best model during the latest step for filtering
#     # for param in filtering_model.parameters():
#     #     param.detach_()
#
#     precision = len(set(clean_idxs_train) & set(labeled_idxs)) / len(labeled_idxs)
#     recall = len(set(clean_idxs_train) & set(labeled_idxs)) / len(clean_idxs_train)
#     iter_filtering += 1
#     print('Filtering Completed! Precision: %.3f Recall: %.3f' % (precision, recall))
#     print('Labeled: %d\tUnlabeled: %d' % (len(labeled_idxs), len(unlabeled_idxs)))
#
# print('Training Completed!')
# print('Testing ...')
# checkpoint = torch.load(path_checkpoint)
# best_model = model_factory(args.pretrained, num_classes)
# best_model = nn.DataParallel(best_model).cuda()
# best_model.load_state_dict(checkpoint['model_weight'])
# sl = Supervised(best_model, optimizer, args)
# best_acc = sl.test(testloader)
#
# print('Best Accuracy: %.3f%%' % best_acc)
#
# name = 'analysis_standard' + str(args.noise_ratio) +'.pickle'
# with open(name, 'wb') as fw:
#     pickle.dump(analysis_dict, fw)