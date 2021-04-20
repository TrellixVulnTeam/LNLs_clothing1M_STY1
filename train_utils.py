import pdb

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import cli
from mean_teacher import ramps, data2
from mean_teacher.data import NO_LABEL
import architectures


args = cli.args
use_cuda = torch.cuda.is_available()


def get_softmax(model, dataloader):
    model.eval()
    batch_size = 0
    dataset = dataloader.dataset
    num_classes = len(set(dataset.targets))
    logits = np.zeros((len(dataset), num_classes))
    with torch.no_grad():
        for batch_idx, ((inputs, _), targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            if batch_idx == 0:
                batch_size = len(targets)
            outputs, _ = model(inputs)
            logits[batch_idx*batch_size:(batch_idx+1)*batch_size] =\
                    outputs.cpu().numpy()
        softmax = logit_to_softmax(logits)
    return softmax


def logit_to_softmax(a):
    c = np.max(a, 1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, 1).reshape(-1, 1)
    y = exp_a / sum_exp_a
    return y


def save_checkpoint(epoch, model, ema_model, swa_model, fastswa_model, accuracy, args, path_checkpoint):
    # Save checkpoint.
    print('\033[33m' + '*****************Saved*****************' + '\033[37m')
    state = {
        'epoch': epoch,
        'model_weight': model.state_dict(),
        'ema_model_weight': ema_model.state_dict(),
        'swa_model_weight': swa_model.state_dict(),
        'fastswa_model_weight': fastswa_model.state_dict(),
        'accuracy': accuracy,
        'args': args
    }
    torch.save(state, path_checkpoint)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,
                                                   args.consistency_rampup)

def lr_fastswa(optimizer, epoch,  # modified for fastSWA
               step_in_epoch, total_steps_in_epoch):
    lr = args.lr # max lr ( initial lr )
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from
    # https://arxiv.org/abs/1706.02677
    # lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr # no rampup when doing fastSWA

    # Cosine LR rampdown from
    # https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.cycle_rampdown_epochs:
        assert args.cycle_rampdown_epochs >= args.epochs
        if epoch <= args.epochs:
            lr *= ramps.cosine_rampdown(epoch, args.cycle_rampdown_epochs)
        else:
            epoch_ = (args.epochs - args.cycle_interval) + ((epoch - args.epochs) % args.cycle_interval)
            lr *= ramps.cosine_rampdown(epoch_, args.cycle_rampdown_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_cosineannealing(optimizer, epoch,  # cosineannealing (variable : half interval (args.interval) )
                                         step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from
    # https://arxiv.org/abs/1706.02677
    # lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from
    # https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.interval:
        # assert args.lr_rampdown_epochs >= args.max_epochs_per_filtering
        lr *= ramps.cosine_rampdown(epoch, args.interval)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, # original code
                         step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from
    # https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from
    # https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.max_epochs_per_filtering
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))


def create_model(ema=False):

    model_factory = architectures.__dict__[args.arch]
    model = model_factory(args.pretrained, args.num_classes)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def moving_average(swa_model, model, alpha=1.0):
    for param1, param2 in zip(swa_model.parameters(), model.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
    return swa_model


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def update_batchnorm(model, train_loader):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        # speeding things up (100 instead of ~800 updates)
        if i > 100:
            return
        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(targets.cuda(non_blocking=True), volatile=True)
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0  # remove to get rid of error in cifar100 w aug
        model_out = model(input_var)


def dataloader_filtering(filtered_trainloader, labeled_idxs, unlabeled_idxs, args):

    batch_sampler = data2.TwoStreamBatchSampler(unlabeled_idxs,
                                               labeled_idxs,
                                               args.batch_size,
                                               args.labeled_batch_size)
    filtered_trainloader = DataLoader(filtered_trainloader.dataset,
                                      batch_sampler=batch_sampler,
                                      num_workers=args.workers,
                                      pin_memory=True)

    return filtered_trainloader