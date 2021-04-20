import torch
import torch.nn as nn
import numpy as np
import pdb

from mean_teacher import losses
from train_utils import get_current_consistency_weight
from train_utils import update_ema_variables
from utils import progress_bar
from lr_sceduler import lr_cosineannealing, lr_fastswa
from sklearn.mixture import GaussianMixture

class Supervised():
    def __init__(self, model, optimizer, args):
        self.seed = args.seed
        self.model = model
        self.optimizer = optimizer
        # train configuration
        self.batch_size = args.batch_size
        self.logit_distance_cost = args.logit_distance_cost
        self.max_total_epochs = args.max_total_epochs

        self.lr_schedule = args.lr_schedule

        self.best_acc = 0
        self.global_step = 0

        self.wait = 0

    def train(self, trainloader, epoch):
        # criterion
        # print('\nEpoch: %d/%d'
        #       % (epoch + 1, self.max_total_epochs))
        class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        residual_logit_criterion = losses.symmetric_mse_loss

        self.model.train()

        running_class_loss = 0
        running_res_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, ((inputs1, _), targets) in enumerate(trainloader):

            if self.lr_schedule == 'fastswa':
                lr_fastswa(self.optimizer, epoch, batch_idx, len(trainloader))
            elif self.lr_schedule == 'cosineannealing':
                lr_cosineannealing(self.optimizer, epoch, batch_idx, len(trainloader))

            inputs1, targets = inputs1.cuda(), targets.cuda()
            outputs = self.model(inputs1)

            minibatch_size = len(targets)

            logit1, logit2 = outputs
            if self.logit_distance_cost >= 0:
                class_logit, cons_logit = logit1, logit2
                res_loss = self.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            else:
                class_logit, cons_logit = logit1, logit1
                res_loss = 0
            class_loss = class_criterion(class_logit,
                                         targets) / minibatch_size

            _, predicted = torch.max(class_logit, 1)
            total += minibatch_size
            correct += predicted.eq(targets).cpu().sum().item()

            loss = class_loss + res_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            running_res_loss += res_loss.item()
            running_class_loss += class_loss.item()
            running_loss += loss.item()

            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | ClassLoss = %.3f | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                         # 'Loss: %.3f | ClassLoss = %.3f | LesLoss: %.3f | clean Acc: %.3f%% (%d/%d) | noisy Acc: %.3f%%(%d/%d)| lr: %.6f'
                         % (running_loss / (batch_idx + 1),
                            running_class_loss / (batch_idx + 1),
                            running_res_loss / (batch_idx + 1),
                            100. * correct / total, correct, total,
                            self.optimizer.param_groups[-1]['lr']))

        loss = {'loss': running_loss / (batch_idx + 1),
                'class_loss': running_class_loss / (batch_idx + 1),
                'res_loss': running_res_loss / (batch_idx + 1)}
        train_acc = 100. * correct / total

        return loss['loss'], train_acc #, clean_acc, noisy_acc

    def validate(self, valloader, epoch):
        self.model.eval()

        running_class_loss = 0
        correct = 0
        total = 0

        class_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = self.model(inputs)
                minibatch_size = len(targets)
                class_loss = class_criterion(outputs, targets) / minibatch_size
                running_class_loss += class_loss.item()
                _, predicted = torch.max(outputs, 1)
                total += minibatch_size
                correct += predicted.eq(targets).cpu().sum().item()
                progress_bar(batch_idx, len(valloader),
                             '[Student]Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (running_class_loss/(batch_idx+1),
                                100.*correct/total, correct, total))
        acc = 100. * correct / total
        loss = running_class_loss/(batch_idx+1)

        return loss, acc

    def test(self, testloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                progress_bar(batch_idx, len(testloader),
                             '[Student]Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))
        acc = 100. * correct / total
        return acc


class MeanTeacher():
    def __init__(self, model, ema_model, optimizer, args):
        self.seed = args.seed
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        # train configuration
        self.batch_size = args.batch_size
        self.labeled_batch_size = args.labeled_batch_size
        self.logit_distance_cost = args.logit_distance_cost
        self.consistency_type = args.consistency_type
        self.ema_decay = args.ema_decay
        self.lr_schedule = args.lr_schedule
        # self.max_total_epochs = args.max_total_epochs
        # self.max_epochs_per_filtering = args.max_epochs_per_filtering

        self.best_ema_acc = 0
        self.global_step = 0
        self.wait = 0

    def train(self, trainloader, epoch):
        class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        if self.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif self.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, self.consistency_type
        residual_logit_criterion = losses.symmetric_mse_loss

        self.model.train()
        self.ema_model.train()

        running_class_loss = 0
        running_consistency_loss = 0
        running_res_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, ((inputs, ema_inputs), targets) in enumerate(trainloader):

            if self.lr_schedule == 'fastswa':
                lr_fastswa(self.optimizer, epoch, batch_idx, len(trainloader))
            elif self.lr_schedule == 'cosineannealing':
                lr_cosineannealing(self.optimizer, epoch, batch_idx, len(trainloader))

            inputs, ema_inputs, targets = inputs.cuda(), ema_inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            ema_outputs = self.ema_model(ema_inputs)

            minibatch_size = len(targets)
            labeled_minibatch_size = torch.sum(targets != -1).item()

            logit1, logit2 = outputs
            ema_logit, _ = ema_outputs
            if self.logit_distance_cost >= 0:
                class_logit, cons_logit = logit1, logit2
                res_loss = self.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            else:
                class_logit, cons_logit = logit1, logit1
                res_loss = 0
            class_loss = class_criterion(class_logit, targets) / minibatch_size
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size

            _, predicted = torch.max(class_logit, 1)
            total += labeled_minibatch_size
            correct += predicted.eq(targets).cpu().sum().item()

            loss = class_loss + consistency_loss + res_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            update_ema_variables(self.model, self.ema_model, self.ema_decay, self.global_step)

            running_res_loss += res_loss.item()
            running_class_loss += class_loss.item()
            running_consistency_loss += consistency_loss.item()
            running_loss += loss.item()

            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | ClassLoss = %.3f | ConsLoss: %.3f | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                         % (running_loss/(batch_idx+1),
                            running_class_loss/(batch_idx+1),
                            running_consistency_loss/(batch_idx+1),
                            running_res_loss/(batch_idx+1),
                            100.*correct/total, correct, total,
                            self.optimizer.param_groups[-1]['lr']))
        loss = {'loss': running_loss / (batch_idx+1),
                'class_loss': running_class_loss / (batch_idx+1),
                'consistency_loss': running_consistency_loss / (batch_idx+1),
                'res_loss': running_res_loss / (batch_idx+1)}
        acc = 100. * correct / total

        return loss['loss'], acc

    def validate(self, valloader, epoch):
        self.model.eval()
        self.ema_model.eval()

        running_class_loss = 0
        running_ema_class_loss = 0
        correct = 0
        ema_correct = 0
        total = 0

        class_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = self.model(inputs)
                ema_outputs, _ = self.ema_model(inputs)
                minibatch_size = len(targets)
                class_loss = class_criterion(outputs, targets) / minibatch_size
                ema_class_loss = class_criterion(ema_outputs, targets) / minibatch_size
                running_class_loss += class_loss.item()
                running_ema_class_loss += ema_class_loss.item()
                _, predicted = torch.max(outputs, 1)
                _, ema_predicted = torch.max(ema_outputs, 1)
                total += minibatch_size
                correct += predicted.eq(targets).cpu().sum().item()
                ema_correct += ema_predicted.eq(targets).cpu().sum().item()
                progress_bar(batch_idx, len(valloader),
                             '[Student]Loss: %.3f | Acc: %.3f%% (%d/%d)  [Teacher]Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (running_class_loss/(batch_idx+1),
                                100.*correct/total, correct, total,
                                running_ema_class_loss/(batch_idx+1),
                                100.*ema_correct/total, ema_correct, total))
        acc = 100. * correct / total
        ema_acc = 100. * ema_correct / total
        if ema_acc >= self.best_ema_acc: # update ema_best_acc & init the 'wait' epochs
            self.best_ema_acc = ema_acc
            self.wait = 0
        else:
            self.wait += 1
        loss = running_class_loss/(batch_idx+1)
        ema_loss = running_ema_class_loss/(batch_idx+1)

        return loss, ema_loss, acc, ema_acc

    def test(self, testloader):
        self.model.eval()
        self.ema_model.eval()
        correct = 0
        ema_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = self.model(inputs)
                ema_outputs, _ = self.ema_model(inputs)
                _, predicted = torch.max(outputs, 1)
                _, ema_predicted = torch.max(ema_outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                ema_correct += ema_predicted.eq(targets).cpu().sum().item()
                progress_bar(batch_idx, len(testloader),
                             '[Student]Acc: %.3f%% (%d/%d)  [Teacher]Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total,
                                100. * ema_correct / total, ema_correct, total))
        acc = 100. * correct / total
        ema_acc = 100. * ema_correct / total
        return acc, ema_acc


class Filter():
    def __init__(self, trainloader, clean_labels, noisy_labels, args):
        self.trainloader = trainloader
        self.trainset = trainloader.dataset
        self.num_classes = args.num_classes
        self.softmax_ema = np.zeros(1)
        self.ema_decay = args.ema_decay
        self.softmax = np.zeros(1)

        self.clean_labels = np.array(clean_labels[0])
        self.noisy_labels = np.array(noisy_labels[0])
        self.labeled_idxs_history = np.array([], dtype=int)

    def filter(self, filtering_model, filtering = False):
        softmax, loss = self._get_softmax(filtering_model, self.trainloader)
        if self.softmax_ema.shape[0] == 1:
            self.softmax_ema = softmax
        else:
            self.softmax_ema = self.ema_decay * self.softmax_ema + (1 - self.ema_decay) * softmax
        pred_ensemble = np.argmax(self.softmax_ema, 1)

        if not filtering:
            return softmax, loss
        else:
            if_agree = (pred_ensemble == self.trainset.targets)
            labeled_idxs = list(np.where(if_agree)[0])
            unlabeled_idxs = list(set(range(len(self.trainset.targets))) - set(labeled_idxs))
            return labeled_idxs, unlabeled_idxs #, softmax, self.softmax_ema, loss

    def filter_(self, model, filtering_type, num_snapshot=0,):

        # get outputs of (last)model
        with torch.no_grad():  ##
            class_criterion = nn.CrossEntropyLoss(reduction='none')
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = model(inputs)
                class_loss = class_criterion(outputs, targets)
                if batch_idx == 0:
                    logits = outputs.cpu().numpy()
                    loss = class_loss.cpu().numpy()
                else:
                    logits = np.concatenate((logits, outputs.cpu().numpy()), axis=0)
                    loss = np.concatenate((loss, class_loss.cpu().numpy()), axis=0)

            softmax = self._logit_to_softmax(logits)

            if filtering_type == 'snapshot_preds_ensemble':
                if num_snapshot == 1:
                    self.softmax = softmax
                else:
                    num = num_snapshot
                    self.softmax = self.softmax * (1-1/num) + softmax * (1/num)
            else: # filtering_type == 'swa_model' or 'fastswa_model'
                self.softmax = softmax

            preds = np.argmax(self.softmax, 1)

        # get consensus for filtering
        clean_or_not = self.clean_labels == self.noisy_labels  # array
        consensus_idxs = preds == self.noisy_labels
        consensus_idxs = np.argwhere(consensus_idxs == True).reshape(-1)
        print('Prec: %.3f%% Num: %d' % (100. * sum(clean_or_not[consensus_idxs]) / len(consensus_idxs),
                                        len(consensus_idxs)))  # precision # len
        consensus_idxs = np.setdiff1d(consensus_idxs, self.labeled_idxs_history)
        if len(consensus_idxs) == 0:
            print('No added samples')
        else:
            print('Consensus Acc: %.3f%%  Num added samples: %d' % (
                100. * sum(clean_or_not[consensus_idxs]) / len(clean_or_not[consensus_idxs]), len(consensus_idxs)))
        return consensus_idxs


    def _get_softmax(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            class_criterion = nn.CrossEntropyLoss(reduction='none')
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _ = model(inputs)
                class_loss = class_criterion(outputs, targets)
                if batch_idx == 0:
                    logits = outputs.cpu().numpy()
                    loss = class_loss.cpu().numpy()
                else:
                    logits = np.concatenate((logits, outputs.cpu().numpy()), axis=0)
                    loss = np.concatenate((loss, class_loss.cpu().numpy()), axis=0)

            softmax = self._logit_to_softmax(logits)
        return softmax, loss

    def _logit_to_softmax(self, a):
        c = np.max(a, 1).reshape(-1, 1)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, 1).reshape(-1, 1)
        y = exp_a / sum_exp_a
        return y
