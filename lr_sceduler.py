from mean_teacher import ramps
import cli

args = cli.args


def lr_cosineannealing(optimizer, epoch, i, len_loader):
    # lr = args.lr
    epoch = epoch + i / len_loader
    epoch = epoch % args.interval
    assert args.lr_schedule == 'cosineannealing'
    assert epoch <= args.interval
    lr = ramps.cosine_rampdown_modified(epoch, args.interval)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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