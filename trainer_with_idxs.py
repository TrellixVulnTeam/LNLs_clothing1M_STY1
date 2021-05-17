import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb

from mean_teacher import losses
from train_utils import get_current_consistency_weight
from train_utils import update_ema_variables
from utils import progress_bar
from torch.autograd import Variable
from copy import deepcopy

class Supervised():

    def __init__(self, model, optimizer, lr_scheduler, args):

        self.seed = args.seed
        self.model = model
        self.optimizer = optimizer

        # train configuration
        self.batch_size = args.batch_size
        self.logit_distance_cost = args.logit_distance_cost
        self.total_epochs = args.interval * args.cycles
        self.args = args

        self.lr_scheduler = lr_scheduler

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

        for batch_idx, ((inputs1, _), targets, indexes) in enumerate(trainloader):

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

        self.lr_scheduler.step()
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
            for batch_idx, (inputs, targets, indexes) in enumerate(valloader):
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
            for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
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

    def __init__(self, model, ema_model, optimizer, lr_scheduler, clean_idxs, clean_labels, noisy_labels, args):
        self.seed = args.seed
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # train configuration
        self.batch_size = args.batch_size
        self.labeled_batch_size = args.labeled_batch_size
        self.logit_distance_cost = args.logit_distance_cost
        self.consistency_type = args.consistency_type
        self.ema_decay = args.ema_decay
        self.total_epochs = args.max_total_epoch

        self.args = args
        self.global_step = 0

        self.clean_idxs = clean_idxs
        self.clean_labels = np.array(clean_labels[0])
        self.noisy_labels = np.array(noisy_labels[0])

    # consistency loss 없이 weight update만
    def warmup(self, trainloader, epoch):

        print('\nEpoch: %d/%d'
              % (epoch+1, self.total_epochs))

        # class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        class_criterion = nn.CrossEntropyLoss(reduction='sum')
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
        correct_t = 0
        correct_clean = 0
        correct_clean_t = 0
        correct_noisy = 0
        correct_noisy_t = 0
        correct_memorized = 0
        correct_memorized_t = 0
        total = 0
        total_clean = 0
        total_noisy = 0

        loss_all = np.zeros(len(trainloader.dataset.targets))
        softmax_all = np.zeros((len(trainloader.dataset.targets), max(trainloader.dataset.targets)+1))
        loss_all_t = np.zeros(len(trainloader.dataset.targets))
        softmax_all_t = np.zeros((len(trainloader.dataset.targets), max(trainloader.dataset.targets)+1))
        for batch_idx, ((inputs, ema_inputs), targets, indexes) in enumerate(trainloader):

            idxs_clean = np.intersect1d(indexes, self.clean_idxs) # sorted, [0,44999]
            clean_or_not = list()
            for index in indexes:
                if index.item() in idxs_clean:
                    clean_or_not.append(True)
                else:
                    clean_or_not.append(False)
            clean_or_not = np.array(clean_or_not)
            clean_targets = self.clean_labels[indexes]
            assert (clean_targets[clean_or_not] == np.array(targets)[clean_or_not]).all()
            inputs, ema_inputs, targets = inputs.cuda(), ema_inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            ema_outputs = self.ema_model(ema_inputs)

            minibatch_size = self.batch_size

            logit1, logit2 = outputs
            ema_logit, _ = ema_outputs
            if self.logit_distance_cost >= 0:
                class_logit, cons_logit = logit1, logit2
                res_loss = self.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            else:
                class_logit, cons_logit = logit1, logit1
                res_loss = 0

            class_loss = class_criterion(class_logit, targets) / minibatch_size
            class_loss_t = class_criterion(ema_logit, targets) / minibatch_size
            # loss_all[indexes] = deepcopy(class_loss.detach().cpu())
            softmax_all[indexes] = deepcopy(F.softmax(class_logit).detach().cpu())
            # loss_all_t[indexes] = deepcopy(class_loss_t.cpu())
            softmax_all_t[indexes] = deepcopy(F.softmax(ema_logit).cpu())
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size

            _, predicted = torch.max(class_logit, 1)
            _, predicted_t = torch.max(ema_logit, 1)
            total += minibatch_size
            total_clean += sum(clean_or_not)
            total_noisy += sum(clean_or_not==False)
            correct += predicted.eq(targets).cpu().sum().item()
            correct_t += predicted_t.eq(targets).cpu().sum().item()
            correct_clean += predicted.eq(targets)[clean_or_not].cpu().sum().item() # clean samples 중 gt label(== noisy label)로 predict 한 개수
            correct_clean_t += predicted_t.eq(targets)[clean_or_not].cpu().sum().item()
            correct_noisy += predicted.cpu().eq(torch.Tensor(clean_targets))[clean_or_not==False].cpu().sum().item() # noisy samples 중 gt label로 predict 한 개수
            correct_noisy_t += predicted_t.cpu().eq(torch.Tensor(clean_targets))[clean_or_not == False].cpu().sum().item()
            correct_memorized += predicted.eq(targets)[clean_or_not==False].cpu().sum().item() # noisy samples 중 noisy label로 predict 한 개수
            correct_memorized_t += predicted_t.eq(targets)[clean_or_not == False].cpu().sum().item()

            # loss = class_loss.mean() + consistency_loss + res_loss
            # loss = class_loss.mean() + res_loss
            loss = class_loss + res_loss + consistency_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            update_ema_variables(self.model, self.ema_model, self.ema_decay, self.global_step)


            running_res_loss += res_loss.item()
            # running_class_loss += class_loss.mean().item()
            running_class_loss += class_loss.item()
            running_consistency_loss += consistency_loss.item()
            running_loss += loss.item()

            progress_bar(batch_idx, len(trainloader),
                         # 'Loss: %.3f | ClassLoss = %.3f | ConsLoss: %.3f | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                         'Loss: %.3f | ClassLoss = %.3f | ConsLoss: %.3f | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | cleanAcc: %.3f%% (%d/%d) | noisyAcc: %.3f%% (%d/%d) | noisyMem: %.3f%% (%d/%d) | lr: %.6f'

                         % (running_loss/(batch_idx+1),
                            running_class_loss/(batch_idx+1),
                            running_consistency_loss/(batch_idx+1),
                            running_res_loss/(batch_idx+1),
                            100. * correct / total, correct, total,
                            100. * correct_clean / total_clean, correct_clean, total_clean,
                            100. * correct_noisy / total_noisy, correct_noisy, total_noisy,
                            100. * correct_memorized / total_noisy, correct_memorized, total_noisy,
                            self.optimizer.param_groups[-1]['lr']))

        acc = 100. * correct / total
        clean_acc = 100. * correct_clean / total_clean
        noisy_acc = 100. * correct_noisy / total_noisy
        noisy_memorized = 100. * correct_memorized / total_noisy
        acc_t = 100. * correct_t / total
        clean_acc_t = 100. * correct_clean_t / total_clean
        noisy_acc_t = 100. * correct_noisy_t / total_noisy
        noisy_memorized_t = 100. * correct_memorized_t / total_noisy
        dict_student = {'train_acc': acc, 'clean_acc':clean_acc, 'noisy_acc':noisy_acc,
                        'noisy_memorized':noisy_memorized, 'softmax':softmax_all}#, 'loss':loss_all,'consistency_loss':[]}
        dict_teacher = {'train_acc': acc_t, 'clean_acc':clean_acc_t, 'noisy_acc':noisy_acc_t,
                        'noisy_memorized':noisy_memorized_t, 'softmax':softmax_all_t}#, 'loss':loss_all_t, 'consistency_loss':[]}

        self.lr_scheduler.step()
        loss = {'loss': running_loss / (batch_idx+1),
                'class_loss': running_class_loss / (batch_idx+1),
                # 'consistency_loss': running_consistency_loss / (batch_idx+1),
                'res_loss': running_res_loss / (batch_idx+1)}


        return loss['loss'], dict_student, dict_teacher


    def train(self, trainloader, epoch):

        print('\nEpoch: %d/%d'
              % (epoch+1, self.total_epochs))

        class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        if self.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss # reduction = 'sum' => 'none'
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
        correct_t = 0
        correct_clean = 0
        correct_clean_t = 0
        correct_noisy = 0
        correct_noisy_t = 0
        correct_memorized = 0
        correct_memorized_t = 0
        total = 0
        total_clean = 0
        total_noisy = 0

        loss_all = np.zeros(len(trainloader.dataset.targets))
        softmax_all = np.zeros((len(trainloader.dataset.targets), max(trainloader.dataset.targets)+1))
        loss_all_t = np.zeros(len(trainloader.dataset.targets))
        softmax_all_t = np.zeros((len(trainloader.dataset.targets), max(trainloader.dataset.targets)+1))
        consistency_all = np.zeros((len(trainloader.dataset.targets), max(trainloader.dataset.targets)+1))
        for batch_idx, ((inputs, ema_inputs), targets, indexes) in enumerate(trainloader):


            idxs_clean = np.intersect1d(indexes, self.clean_idxs) # sorted, [0,44999]
            clean_or_not = list()
            for index in indexes:
                if index.item() in idxs_clean: # idxs_clean : 해당 batch 내에서 실제 clean samples인 samples의 indexes
                    clean_or_not.append(True)
                else:
                    clean_or_not.append(False)
            clean_or_not = np.array(clean_or_not) # 해당 batch 내의 각 sample들이 clean인지 아닌지 (bool list)
            clean_targets = self.clean_labels[indexes]
            noisy_targets = self.noisy_labels[indexes]
            # assert (clean_targets[clean_or_not] == targets[clean_or_not]).all() # clean samples 의 target이 -1일수도 있어서 train 할때는 이 과정 생략

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

            class_loss = class_criterion(class_logit, targets) / labeled_minibatch_size
            class_loss_t = class_criterion(ema_logit, targets) / labeled_minibatch_size
            # loss_all[indexes] = deepcopy(class_loss.detach().cpu())
            softmax_all[indexes] = deepcopy(F.softmax(class_logit).detach().cpu())
            # loss_all_t[indexes] = deepcopy(class_loss_t.cpu())
            softmax_all_t[indexes] = deepcopy(F.softmax(ema_logit).cpu())

            consistency_weight = get_current_consistency_weight(epoch-self.args.warmup_epoch)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            # consistency_all[indexes] = deepcopy(consistency_loss.detach().cpu())


            _, predicted = torch.max(class_logit, 1)
            _, predicted_t = torch.max(ema_logit, 1)
            total += labeled_minibatch_size
            total_clean += sum(clean_or_not) # labeled + unlabeled => 45000 * (1-noise_rate)
            total_noisy += sum(clean_or_not==False) # labeled + unlabeled => 45000 * noise_rate
            correct += predicted.eq(targets).cpu().sum().item() # only labeled ( for train_acc )
            correct_t += predicted_t.eq(targets).cpu().sum().item() # only labeled ( for train_acc )
            correct_clean += predicted.cpu().eq(torch.Tensor(clean_targets))[clean_or_not].cpu().sum().item() # clean samples 중 gt label(== noisy label)로 predict 한 개수
            correct_clean_t += predicted_t.cpu().eq(torch.Tensor(clean_targets))[clean_or_not].cpu().sum().item()
            correct_noisy += predicted.cpu().eq(torch.Tensor(clean_targets))[clean_or_not==False].cpu().sum().item() # noisy samples 중 gt label로 predict 한 개수
            correct_noisy_t += predicted_t.cpu().eq(torch.Tensor(clean_targets))[clean_or_not == False].cpu().sum().item()
            correct_memorized += predicted.cpu().eq(torch.Tensor(noisy_targets))[clean_or_not==False].cpu().sum().item() # noisy samples 중 noisy label로 predict 한 개수
            correct_memorized_t += predicted_t.cpu().eq(torch.Tensor(noisy_targets))[clean_or_not == False].cpu().sum().item()

            # loss = class_loss.mean() + consistency_loss.mean() + res_loss
            loss = class_loss + consistency_loss + res_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            update_ema_variables(self.model, self.ema_model, self.ema_decay, self.global_step)

            running_res_loss += res_loss.item()
            running_class_loss += class_loss.mean().item()
            # running_consistency_loss += consistency_loss.mean().item()
            running_consistency_loss += consistency_loss.item()
            running_loss += loss.item()

            progress_bar(batch_idx, len(trainloader),
                         # 'Loss: %.3f | ClassLoss = %.3f | ConsLoss: %.3f | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                         'Loss: %.3f | ClassLoss = %.3f | ConsLoss: %.3f  | LesLoss: %.3f | Acc: %.3f%% (%d/%d) | cleanAcc: %.3f%% (%d/%d) | noisyAcc: %.3f%% (%d/%d) | noisyMem: %.3f%% (%d/%d) | lr: %.6f'
                         % (running_loss/(batch_idx+1),
                            running_class_loss/(batch_idx+1),
                            running_consistency_loss/(batch_idx+1),
                            running_res_loss/(batch_idx+1),
                            100. * correct / total, correct, total,
                            100. * correct_clean / total_clean, correct_clean, total_clean,
                            100. * correct_noisy / total_noisy, correct_noisy, total_noisy,
                            100. * correct_memorized / total_noisy, correct_memorized, total_noisy,
                            self.optimizer.param_groups[-1]['lr']))

        assert total_clean + total_noisy
        acc = 100. * correct / total
        clean_acc = 100. * correct_clean / total_clean
        noisy_acc = 100. * correct_noisy / total_noisy
        noisy_memorized = 100. * correct_memorized / total_noisy
        acc_t = 100. * correct_t / total
        clean_acc_t = 100. * correct_clean_t / total_clean
        noisy_acc_t = 100. * correct_noisy_t / total_noisy
        noisy_memorized_t = 100. * correct_memorized_t / total_noisy
        dict_student = {'train_acc': acc, 'clean_acc':clean_acc, 'noisy_acc':noisy_acc,
                        'noisy_memorized':noisy_memorized, 'softmax':softmax_all}#, 'loss':loss_all, 'consistency_loss':consistency_all}
        dict_teacher = {'train_acc': acc_t, 'clean_acc':clean_acc_t, 'noisy_acc':noisy_acc_t,
                        'noisy_memorized':noisy_memorized_t, 'softmax':softmax_all_t}#, 'loss':loss_all_t,  'consistency_loss':consistency_all}

        self.lr_scheduler.step()
        loss = {'loss': running_loss / (batch_idx+1),
                'class_loss': running_class_loss / (batch_idx+1),
                'consistency_loss': running_consistency_loss / (batch_idx+1),
                'res_loss': running_res_loss / (batch_idx+1)}
        acc = 100. * correct / total
        # print('Consistency loss: ', loss['consistency_loss'])
        return loss['loss'], dict_student, dict_teacher

    def validate(self, valloader):
        self.model.eval()
        self.ema_model.eval()

        running_class_loss = 0
        running_ema_class_loss = 0
        correct = 0
        ema_correct = 0
        total = 0

        class_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (inputs, targets, indexes) in enumerate(valloader):
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
            for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
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
        self.num_classes = len(self.trainset.targets)
        self.softmax_ema = np.zeros(1)
        self.t_softmax_ema = np.zeros(1)
        self.ema_decay = args.ema_decay

        self.clean_labels = np.array(clean_labels[0])
        self.noisy_labels = np.array(noisy_labels[0])
        self.clean_or_not = self.clean_labels == self.noisy_labels
        self.noisy_or_not = self.clean_labels != self.noisy_labels
        self.labeled_idxs_history = np.array([], dtype=int)

        self.consistency_type = args.consistency_type
        self.logit_distance_cost = args.logit_distance_cost


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


    def get_behavior_mt(self, model, ema_model, filtering = False):

        softmax, loss, t_softmax, t_loss, loss_consistency = self._get_softmax_mt(model, ema_model, self.trainloader)
        if self.softmax_ema.shape[0] == 1:
            self.softmax_ema = softmax
            self.t_softmax_ema = t_softmax
        else:
            self.softmax_ema = self.ema_decay * self.softmax_ema + (1 - self.ema_decay) * softmax
            self.t_softmax_ema = self.ema_decay * self.t_softmax_ema + (1 - self.ema_decay) * t_softmax

        ensemble_preds = np.argmax(self.softmax_ema, 1)
        preds = np.argmax(softmax, 1)
        t_preds = np.argmax(t_softmax, 1)

        assert sum(self.clean_or_not) + sum(self.noisy_or_not) == len(preds)

        clean_acc = 100.*sum(preds[self.clean_or_not]==self.clean_labels[self.clean_or_not])/sum(self.clean_or_not)
        noisy_acc = 100.*sum(preds[self.noisy_or_not]==self.clean_labels[self.noisy_or_not])/sum(self.noisy_or_not)
        noisy_memorized = 100. * sum(preds[self.noisy_or_not] == self.noisy_labels[self.noisy_or_not]) / sum(self.noisy_or_not)
        t_clean_acc = 100. * sum(t_preds[self.clean_or_not] == self.clean_labels[self.clean_or_not]) / sum(self.clean_or_not)
        t_noisy_acc = 100. * sum(t_preds[self.noisy_or_not] == self.clean_labels[self.noisy_or_not]) / sum(self.noisy_or_not)
        t_noisy_memorized = 100. * sum(t_preds[self.noisy_or_not] == self.noisy_labels[self.noisy_or_not]) / sum(self.noisy_or_not)

        student_data = {'clean_acc':clean_acc, 'noisy_acc':noisy_acc, 'noisy_memorized':noisy_memorized,
                        'softmax':softmax, 'softmax_ema':self.softmax_ema, 'loss':loss}
        teacher_data = {'clean_acc':t_clean_acc, 'noisy_acc': t_noisy_acc, 'noisy_memorized': t_noisy_memorized,
                        'softmax':t_softmax, 'softmax_ema':self.t_softmax_ema, 'loss':t_loss}
        return student_data, teacher_data, loss_consistency


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

    def _get_softmax_mt(self, model, ema_model, dataloader):
        model.eval()
        ema_model.eval()
        with torch.no_grad():
            class_criterion = nn.CrossEntropyLoss(reduction='none')
            if self.consistency_type == 'mse':
                consistency_criterion = losses.softmax_mse_loss
            elif self.consistency_type == 'kl':
                consistency_criterion = losses.softmax_kl_loss
            else:
                assert False, self.consistency_type

            # assert input_logits.size() == target_logits.size()
            # input_softmax = F.softmax(input_logits, dim=1)
            # target_softmax = F.softmax(target_logits, dim=1)
            # num_classes = input_logits.size()[1]
            # return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes

            for batch_idx, ((inputs, ema_inputs), targets) in enumerate(dataloader):

                inputs, ema_inputs, targets = inputs.cuda(), ema_inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                ema_outputs = ema_model(ema_inputs)

                minibatch_size = len(targets)

                logit1, logit2 = outputs
                ema_logit, _ = ema_outputs
                if self.logit_distance_cost >= 0:
                    class_logit, cons_logit = logit1, logit2
                else:
                    class_logit, cons_logit = logit1, logit1
                class_loss = class_criterion(class_logit, targets)
                t_class_loss = class_criterion(ema_logit, targets)
                # consistency_loss = consistency_criterion(cons_logit, ema_logit) / minibatch_size
                input_softmax = F.softmax(cons_logit, dim=1)
                target_softmax = F.softmax(ema_logit, dim=1)
                consistency_loss = F.mse_loss(input_softmax, target_softmax, reduction='none').sum(axis=1)

                if batch_idx == 0:
                    logits = class_logit.cpu().numpy()
                    loss = class_loss.cpu().numpy()
                    t_logits = ema_logit.cpu().numpy()
                    t_loss = t_class_loss.cpu().numpy()
                    loss_consistency = consistency_loss.cpu().numpy()
                else:
                    logits = np.concatenate((logits, class_logit.cpu().numpy()), axis=0)
                    loss = np.concatenate((loss, class_loss.cpu().numpy()), axis=0)
                    t_logits = np.concatenate((t_logits, ema_logit.cpu().numpy()), axis=0)
                    t_loss = np.concatenate((t_loss, t_class_loss.cpu().numpy()), axis=0)
                    loss_consistency = np.concatenate((loss_consistency, consistency_loss.cpu().numpy()), axis=0)

            softmax = self._logit_to_softmax(logits)
            t_softmax = self._logit_to_softmax(t_logits)
        return softmax, loss, t_softmax, t_loss, loss_consistency

    def _logit_to_softmax(self, a):
        c = np.max(a, 1).reshape(-1, 1)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a, 1).reshape(-1, 1)
        y = exp_a / sum_exp_a
        return y