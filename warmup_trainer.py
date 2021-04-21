import torch
import torch.nn as nn
import numpy as np
import pdb

from mean_teacher import losses
from train_utils import get_current_consistency_weight
from train_utils import update_ema_variables
from utils import progress_bar
from train_utils import lr_cosineannealing
from torch.autograd import Variable

class Supervised():

    def __init__(self, model, optimizer, args):

        self.seed = args.seed
        self.model = model
        self.optimizer = optimizer

        # train configuration
        self.batch_size = args.batch_size
        self.logit_distance_cost = args.logit_distance_cost
        self.total_epochs = args.interval * args.cycles
        self.args = args

    def train(self, trainloader, epoch):
        # criterion
        print('\nEpoch: %d/%d'
              % (epoch + 1, self.total_epochs))
        class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        residual_logit_criterion = losses.symmetric_mse_loss

        self.model.train()

        running_class_loss = 0
        running_res_loss = 0
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, ((inputs1, _), targets) in enumerate(trainloader):

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
                         % (running_loss / (batch_idx + 1),
                            running_class_loss / (batch_idx + 1),
                            running_res_loss / (batch_idx + 1),
                            100. * correct / total, correct, total,
                            self.optimizer.param_groups[-1]['lr']))

        loss = {'loss': running_loss / (batch_idx + 1),
                'class_loss': running_class_loss / (batch_idx + 1),
                'res_loss': running_res_loss / (batch_idx + 1)}
        train_acc = 100. * correct / total

        return loss['loss'], train_acc


    def validate(self, valloader):
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


class Filter():

    def __init__(self, trainloader, clean_labels, noisy_labels, args):

        self.trainloader = trainloader
        self.trainset = trainloader.dataset
        self.num_classes = len(self.trainset.targets)
        self.softmax_ema = np.zeros(1)
        self.ema_decay = args.ema_decay

        self.clean_labels = np.array(clean_labels[0])
        self.noisy_labels = np.array(noisy_labels[0])
        self.clean_or_not = self.clean_labels == self.noisy_labels
        self.noisy_or_not = self.clean_labels != self.noisy_labels
        self.labeled_idxs_history = np.array([], dtype=int)


    def get_behavior(self, model, filtering = False):

        softmax, loss = self._get_softmax(model, self.trainloader)
        if self.softmax_ema.shape[0] == 1:
            self.softmax_ema = softmax
        else:
            self.softmax_ema = self.ema_decay * self.softmax_ema + (1 - self.ema_decay) * softmax

        ensemble_preds = np.argmax(self.softmax_ema, 1)
        preds = np.argmax(softmax, 1)

        assert sum(self.clean_or_not) + sum(self.noisy_or_not) == len(preds)

        clean_acc = 100.*sum(preds[self.clean_or_not]==self.clean_labels[self.clean_or_not])/sum(self.clean_or_not)
        noisy_acc = 100.*sum(preds[self.noisy_or_not]==self.clean_labels[self.noisy_or_not])/sum(self.noisy_or_not)
        noisy_memorized = 100. * sum(preds[self.noisy_or_not] == self.noisy_labels[self.noisy_or_not]) / sum(self.noisy_or_not)

        return softmax, self.softmax_ema, loss, clean_acc, noisy_acc, noisy_memorized


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
