import os
import sys
import time
import smtplib
from email.mime.text import MIMEText
from copy import deepcopy
import matplotlib.pyplot as plt


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %5s' % format_time(step_time))
    L.append(' | Tot: %8s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def send_email(message):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('jihoo94@gmail.com', 'rsmfdyayddpvorpc')

    msg = MIMEText('ㅇㅇ')
    msg['Subject'] = message

    s.sendmail("jihoo94@gmail.com", "jhkim@spa.hanyang.ac.kr", msg.as_string())
    s.quit()


def logging_dict(accuracy, analysis_dict, dict_student, dict_teacher):

    accuracy['train_acc'].append(dict_student['train_acc'])
    # accuracy['clean_acc'].append(dict_student['clean_acc'])
    # accuracy['noisy_acc'].append(dict_student['noisy_acc'])
    # accuracy['noisy_memorized'].append(dict_student['noisy_memorized'])
    accuracy['train_acc_t'].append(dict_teacher['train_acc'])
    # accuracy['clean_acc_t'].append(dict_teacher['clean_acc'])
    # accuracy['noisy_acc_t'].append(dict_teacher['noisy_acc'])
    # accuracy['noisy_memorized_t'].append(dict_teacher['noisy_memorized'])

    analysis_dict['softmax'].append(deepcopy(dict_student['softmax']))
    # analysis_dict['softmax_ema'].append(deepcopy(dict_student['softmax_ema']))
    # analysis_dict['loss_history'].append(deepcopy(dict_student['loss']))
    analysis_dict['softmax_t'].append(deepcopy(dict_teacher['softmax']))
    # analysis_dict['t_softmax_ema'].append(deepcopy(dict_teacher['softmax_ema']))
    # analysis_dict['loss_history_t'].append(deepcopy(dict_teacher['loss']))

    # if len(dict_student['consistency_loss']) != 0:
    #     analysis_dict['consistency_loss'].append(dict_student['consistency_loss'])

    return accuracy, analysis_dict


def plotting(epoch, accuracy, lr, path_plot):

    # plot
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    import pdb;pdb.set_trace()
    ax1.plot(range(epoch), accuracy['train_acc'],
             color='blue', label='Train_Acc')
    ax1.plot(range(epoch), accuracy['val_acc'],
             color='orange', label='Valid_Student_Acc')
    ax1.plot(range(epoch), accuracy['val_ema_acc'],
             color='orange', label='Valid_Teacher_Acc', linestyle=':')
    ax1.plot(range(epoch), accuracy['test_acc'],
             color='green', label='Test_Student_Acc')
    ax1.plot(range(epoch), accuracy['test_ema_acc'],
             color='green', label='Test_Teacher_Acc', linestyle=':')
    # ax1.plot(range(epoch), accuracy['clean_acc'],
    #          color='midnightblue', label='clean_acc')
    # ax1.plot(range(epoch), accuracy['clean_acc_t'],
    #          color='midnightblue', label='clean_acc_t', linestyle=':')
    # ax1.plot(range(epoch), accuracy['noisy_acc'],
    #          color='cornflowerblue', label='noisy_acc')
    # ax1.plot(range(epoch), accuracy['noisy_acc_t'],
    #          color='cornflowerblue', label='noisy_acc_t', linestyle=':')
    # ax1.plot(range(epoch), accuracy['noisy_memorized'],
    #          color='darkred', label='Noisy_Memorized')
    # ax1.plot(range(epoch), accuracy['noisy_memorized_t'],
    #          color='darkred', label='noisy_memorized_t', linestyle=':')
    plt.plot(range(epoch), accuracy['precision'],
             color='saddlebrown', label='Precision', linestyle='--')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('LR')
    ax2.plot(range(epoch), lr,
             color='dimgray', label='LR', linestyle='--')
    ax2.legend(loc='upper right')

    fig.savefig(path_plot, dpi=300)
    plt.close()


def count_per_class(filtered_noisy_labels): # array

    num_class = max(filtered_noisy_labels)
    count_list = list()
    for class_ in range(num_class):
        count_list.append(list(filtered_noisy_labels).count(class_))

    print('num samples per class : ', count_list)