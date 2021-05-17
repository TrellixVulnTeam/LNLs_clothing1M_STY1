import csv,os,random,time
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import datasets_with_indexes
from mean_teacher import data, losses
from utils_clothing1M import send_email, logging_dict, plotting, count_per_class
from noise import generate_noise
import cli
from train_utils import save_checkpoint, create_model, get_filtered_trainloader, filtering, filtering_ssl
from trainer_with_idxs_clothing1M import Supervised, Filter, MeanTeacher
import architectures
import pickle


# set working directory to current folder
p = Path(__file__).absolute()
PATH = p.parents[0]
os.chdir(PATH)
args = cli.args
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dict_analysis = {'softmax':[],
                 'noisy_labels':[],
                 'loss_history':[],
                 'softmax_t':[],
                 'loss_history_t':[],
                 'lr':[],
                 'args':args,
                 'labeled_idxs':[],
                 'consistency_loss':[]}


"""
################################# Prepare Data ###############################
"""

print('==> Preparing data..')

trainset, valset, testset, num_classes =\
        datasets_with_indexes.__dict__[args.dataset](args.final_run, args.train_size)
dict_analysis['noisy_labels'].append(deepcopy(trainset.targets)) # len : 64000

filtered_trainset = deepcopy(trainset)

trainloader_mt = DataLoader(deepcopy(trainset),
                         batch_size=64,
                         shuffle=False,
                         num_workers=args.workers,
                         pin_memory=True)
filtered_trainloader = DataLoader(filtered_trainset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  pin_memory=True)
trainloader = DataLoader(trainset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.workers,              
                         pin_memory=True)
valloader = DataLoader(valset,
                       batch_size=64,
                       shuffle=False,
                       num_workers=args.workers,
                       pin_memory=True)
testloader = DataLoader(testset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=True)

# for saving dataset
data_dict = {'filtered_trainloader':filtered_trainloader,
             'trainloader':trainloader,
             'trainloader_mt':trainloader_mt,
             'valloader':valloader,
             'testloader':testloader,
             'noisy_labels':dict_analysis['noisy_labels'],
             'num_classes':num_classes}

"""
########################### Initialize Model ##################################
"""

print('==> Building model..')
model = create_model(args.pretrained,num_classes)
ema_model = create_model(args.pretrained,num_classes, ema=True)

name_exp = '_'.join([args.arch, args.dataset, args.noise_type,
                    str(args.noise_ratio), 'ema_decay', str(args.ema_decay), 'lr_max', str(args.lr_max),
                     'lr_scheduler', str(args.lr_scheduler), 'lr_interval', str(args.lr_interval),
                    'model_state', str(args.model_state), 'seed', str(args.seed)])

tm = time.localtime(time.time())
#string = str(tm.tm_mon)+'.'+str(tm.tm_mday)+'.'+str(tm.tm_hour)+'.'+str(args.lr)+'.'+str(args.batch_size)+'.'+str(args.decay)+'.'+str(args.interval)
string = str(tm.tm_mon)+'.'+str(tm.tm_mday)+'.'+str(tm.tm_hour)+'.'+str(args.SSL)
directory = os.path.join("./results",string,'warmup')

# make directories
os.makedirs(directory+'/cyclic_ssl/checkpoint', exist_ok=True)
os.makedirs(directory+'/cyclic_ssl/log', exist_ok=True)
os.makedirs(directory+'/cyclic_ssl/plot', exist_ok=True)
os.makedirs(directory+'/cyclic_ssl/analysis', exist_ok=True)

path_checkpoint_best = (directory+'/cyclic_ssl/checkpoint/' + name_exp + 'best')
path_checkpoint_worst = (directory+'/cyclic_ssl/checkpoint/' + name_exp + 'worst')
path_log = (directory+'/cyclic_ssl/log/' + name_exp + '.csv')
path_plot = (directory+'/cyclic_ssl/plot/' + name_exp + '.png')
path_analysis = (directory+'/cyclic_ssl/analysis/' + name_exp + '.pickle')
path_dataset = (directory+'/cyclic_ssl/data_dict.pickle')
with open(path_dataset, 'wb') as fw:
    pickle.dump(data_dict, fw)

cudnn.benchmark = True
print('Number of GPUs:', torch.cuda.device_count())
print('Using CUDA..')


"""
################################ Main #########################################
"""

with open(path_log, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc',
                        'valid student loss', 'valid student acc', 'val teacher loss','val teacher acc',
                        'test student acc', 'test teacher acc'])

f_mt = Filter(trainloader_mt, dict_analysis['noisy_labels'], args)
accuracy = {'train_acc': [],
            'val_acc': [],
            'val_ema_acc': [],
            'test_acc': [],
            'test_ema_acc': [],
            'train_acc_t': []
            }

optimizer = optim.SGD(model.parameters(), lr=args.lr_max,
                      momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)

print('Scheduler type : %s' % (str(args.lr_scheduler)))
print('Cycle interval : %s' % (str(args.lr_interval)))
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= args.lr_interval, T_mult=1, eta_min=args.lr_min)

trainer = MeanTeacher(model, ema_model, optimizer, lr_scheduler,dict_analysis['noisy_labels'], args)
epoch = args.start_epoch

if args.SSL == False:

    while epoch < args.warmup_epoch: # 100

        dict_analysis['lr'].append(trainer.optimizer.param_groups[0]['lr'])
        train_loss, dict_student, dict_teacher = trainer.warmup(filtered_trainloader, epoch)
        val_loss, val_ema_loss, val_acc, val_ema_acc = trainer.validate(valloader)
        test_acc, test_ema_acc = trainer.test(testloader)
        accuracy['val_acc'].append(val_acc)
        accuracy['val_ema_acc'].append(val_ema_acc)
        accuracy['test_acc'].append(test_acc)
        accuracy['test_ema_acc'].append(test_ema_acc)

        epoch += 1
        accuracy, dict_analysis = logging_dict(accuracy, dict_analysis, dict_student, dict_teacher)
        plotting(epoch, accuracy, dict_analysis['lr'], path_plot)

        # log
        with open(path_log, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, dict_student['train_acc'],
                                val_loss, val_acc, val_ema_loss, val_ema_acc,
                                test_acc, test_ema_acc])

        if val_ema_acc == max(accuracy['val_ema_acc']): # when less memorized & almost best accuracy on noisy dataset
            print('**************** Saved as best point *****************')
            state = {'epoch': epoch -1,
                    'best_model_weight': trainer.model.state_dict(),
                    'best_ema_model_weight':trainer.ema_model.state_dict()}
            best_epoch = epoch -1
            torch.save(state, path_checkpoint_best)

else:
    epoch = 100
    with open(path_analysis, 'rb') as fw:
        dict_analysis = pickle.load(fw)
    with open(path_dataset, 'rb') as fw:
        data_dict = pickle.load(fw)
    accuracy = dict_analysis['accuracy']
    args = dict_analysis['args']
    best_epoch = accuracy['val_ema_acc'].index(max(accuracy['val_ema_acc']))

print('Finish warmup!')
dict_analysis['accuracy'] = accuracy
with open(path_analysis, 'wb') as fw:
    pickle.dump(dict_analysis, fw)

print('Start filtering!')
idxs_filtered_s = filtering(dict_analysis['softmax'], best_epoch, deepcopy(trainset.targets))
idxs_filtered_t = filtering(dict_analysis['softmax_t'], best_epoch, deepcopy(trainset.targets))
print('Num filtered by s : %d   Num filtered by t : %d' %(len(idxs_filtered_s), len(idxs_filtered_t)))
targets = trainloader.dataset.targets
labeled_idxs = idxs_filtered_t
unlabeled_idxs = np.setdiff1d(np.arange(len(targets)), labeled_idxs)
filtered_trainloader.dataset.targets = \
    [-1 if i in unlabeled_idxs else targets[i] for i in range(len(targets))]
print('Labeled: %d\tUnlabeled: %d' % (len(labeled_idxs), len(unlabeled_idxs)))
filtered_trainloader = get_filtered_trainloader(filtered_trainloader, labeled_idxs, unlabeled_idxs, args)
dict_analysis['labeled_idxs'].append(labeled_idxs)

# select model state
if args.model_state == 'initialize':
    print('Start with initialized model')
    model = create_model(args.pretrained,num_classes)
    ema_model = create_model(args.pretrained,num_classes,ema=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)

elif args.model_state == 'continue':
    print('Start with last model continuously')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)

elif args.model_state == 'load_best':
    print('Start with best model')
    checkpoint = torch.load(path_checkpoint_best)
    model = create_model(args.pretrained,num_classes)
    ema_model = model = create_model(args.pretrained,num_classes,ema=True)
    model.load_state_dict(checkpoint['best_model_weight'])
    ema_model.load_state_dict(checkpoint['best_ema_model_weight'])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)

if args.if_ssl_cyclic:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_interval_ssl, T_mult=1)

else:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_total_epoch - args.warmup_epoch, eta_min=0.001)

# trainer = MeanTeacher(model, ema_model, optimizer, lr_scheduler, clean_idxs, dict_analysis['clean_labels'], dict_analysis['noisy_labels'], args)
trainer = MeanTeacher(model, ema_model, optimizer, lr_scheduler,dict_analysis['noisy_labels'], args)
noisy_labels = np.array(data_dict['noisy_labels'][0])
prediction_set_s = list()
prediction_set_t = list()
while epoch < args.max_total_epoch: # 300

    if (epoch - args.warmup_epoch) % args.lr_interval_ssl == 0 and len(prediction_set_t) >= 3:
        preds_s = np.argmax(np.array(prediction_set_s)[-3:], axis=2)
        preds_t = np.argmax(np.array(prediction_set_t)[-3:], axis=2)

        idxs_filtered_list_s = list()
        for pred in preds_s:
            idxs = np.argwhere(pred == noisy_labels).reshape(-1)
            idxs_filtered_list_s.append(idxs)
        for i, idxs in enumerate(idxs_filtered_list_s):
            if i == 0:
                idxs_filtered_s = idxs  # array
            else:
                idxs_filtered_s = np.intersect1d(idxs_filtered_s, idxs)
        idxs_filtered_list_t = list()
        for pred in preds_t:
            idxs = np.argwhere(pred == noisy_labels).reshape(-1)
            idxs_filtered_list_t.append(idxs)
        for i, idxs in enumerate(idxs_filtered_list_t):
            if i == 0:
                idxs_filtered_t = idxs  # array
            else:
                idxs_filtered_t = np.intersect1d(idxs_filtered_t, idxs)

        print('Start filtering!')
        print('Best epoch is: ', best_epoch)

        print('Num filtered by s : %d   Num filtered by t : %d' % (len(idxs_filtered_s), len(idxs_filtered_t)))
        print('%d + %d'%(len(labeled_idxs), len(np.intersect1d(unlabeled_idxs, idxs_filtered_t))))
        targets = trainloader.dataset.targets
        labeled_idxs = np.union1d(labeled_idxs, np.intersect1d(unlabeled_idxs, idxs_filtered_t)) # update labeled set
        unlabeled_idxs = np.setdiff1d(np.arange(len(targets)), labeled_idxs)
        filtered_trainloader.dataset.targets = \
            [-1 if i in unlabeled_idxs else targets[i] for i in range(len(targets))]

        # print('Filtering Completed! Precision: %.3f %% Recall: %.3f %%' % (precision * 100, recall * 100))
        print('Labeled: %d\tUnlabeled: %d' % (len(labeled_idxs), len(unlabeled_idxs)))
        filtered_trainloader = get_filtered_trainloader(filtered_trainloader, labeled_idxs, unlabeled_idxs, args)
        dict_analysis['labeled_idxs'].append(labeled_idxs)

        # select model state
        if args.model_state == 'initialize':
            print('Start with initialized model')
            model = create_model(num_classes)
            ema_model = create_model(num_classes, ema=True)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.decay, nesterov=args.nesterov)

        elif args.model_state == 'continue':
            print('Start with last model continuously')
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.decay, nesterov=args.nesterov)

        elif args.model_state == 'load_best':
            print('Start with best model')
            checkpoint = torch.load(path_checkpoint_best)
            model = create_model(num_classes)
            ema_model = create_model(num_classes, ema=True)
            model.load_state_dict(checkpoint['best_model_weight'])
            ema_model.load_state_dict(checkpoint['best_ema_model_weight'])

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.decay, nesterov=args.nesterov)

        if args.if_ssl_cyclic:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_interval_ssl, T_mult=1)
        else:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=args.max_total_epoch - args.warmup_epoch,
                                                                eta_min=0.001)

        trainer = MeanTeacher(model, ema_model, optimizer, lr_scheduler, dict_analysis['noisy_labels'], args)
    # check lr for every epoch
    dict_analysis['lr'].append(trainer.optimizer.param_groups[0]['lr'])
    train_loss, dict_student, dict_teacher = trainer.train(filtered_trainloader, epoch)
    val_loss, val_ema_loss, val_acc, val_ema_acc = trainer.validate(valloader)
    test_acc, test_ema_acc = trainer.test(testloader)
    accuracy['val_acc'].append(val_acc)
    accuracy['val_ema_acc'].append(val_ema_acc)
    accuracy['test_acc'].append(test_acc)
    accuracy['test_ema_acc'].append(test_ema_acc)

    epoch += 1
    if (epoch - args.warmup_epoch) % args.lr_interval_ssl == 0:
        prediction_set_s.append(dict_student['softmax'])
        prediction_set_t.append(dict_teacher['softmax'])
    accuracy, dict_analysis = logging_dict(accuracy, dict_analysis, dict_student, dict_teacher)
    plotting(epoch, accuracy, dict_analysis['lr'], path_plot)
    

    # log
    with open(path_log, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, dict_student['train_acc'],
                            val_loss, val_acc, val_ema_loss, val_ema_acc,
                            test_acc, test_ema_acc])

    if val_ema_acc == max(accuracy['val_ema_acc']): # when less memorized & almost best accuracy on noisy dataset
        print('**************** Saved as best point *****************')
        state = {'epoch': epoch -1,
                 'best_model_weight': trainer.model.state_dict(),
                 'best_ema_model_weight':trainer.ema_model.state_dict()}
        best_epoch = epoch -1
        torch.save(state, path_checkpoint_best)

dict_analysis['accuracy'] = accuracy
with open(path_analysis, 'wb') as fw:
    pickle.dump(dict_analysis, fw)
