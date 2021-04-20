import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--no-filtering', action='store_true', help='no filtering')
# parser.add_argument('--test', action='store_true', help='for debugging')
# parser.add_argument('--hold', action='store_true', help='if true, pdb.set_trace() after training')
# parser.add_argument('--name', default='', type=str, help='name of experiment(you don\'t have to fill)')
# parser.add_argument('--num-labels', default=None, type=int, help='number of labeled indexes')
parser.add_argument('--num-classes', default=None, type=int, help='number of classes')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'clothes1M', 'webvision'], help='choose dataset')
parser.add_argument('--val-size', default=5000, type=int, help='size of validation set')
# parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint')
parser.add_argument('--arch', default="cifar_shakeshake26", type=str, help='model architecture')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='use nesterov momentum')
parser.add_argument('--ema-decay', default=0.97, type=float, help='ema variable decay rate (default: 0.999)')
parser.add_argument('--logit-distance-cost', default=.01, type=float,
                    help='let the student model have two outputs and use an MSE loss between the logits with\
                     the given weight (default: only have one output)')
parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
# parser.add_argument('--checkpoint-epochs', default=20, type=int,
#                     help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
# parser.add_argument('--evaluation-epochs', default=1, type=int,
#                     help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
# parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on test set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# parser.add_argument('--email', action='store_true', help='send email when training is finished')
# parser.add_argument('--final-run', action='store_true', help='train without validation set')
# parser.add_argument('--patience', default=30, type=int, help='patience for early stopping')


# parser.add_argument('--sigua-weight', default=0.001, type=float, help='weight of sigua loss')
# parser.add_argument('--filtering-type', default='small-loss', type=str,
#                     choices=['small-loss', 'softmax_ema', 'gmm', 'confidence'], help='filtering criterion')

# noise
parser.add_argument('--noise-type', default='symmetric', choices=['symmetric', 'asymmetric'], help='type of noise')
parser.add_argument('--noise-ratio', default=0.4, type=float, help='ratio of noise')
parser.add_argument('--noisy-validation', default=True, type=bool, help='clean validation or noisy validation set')

# lr schedule
# parser.add_argument('--c', default=80, type=int, help='interval of lr cycle in fastSWA')
parser.add_argument('--lr-schedule', default='cosineannealing', type=str, help='lr scheduling type', choices=['cosineannealing', 'fastswa'])
parser.add_argument('--lr', default=0.05, type=float, help='max learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float, help='initial learning rate when using linear rampup')
parser.add_argument('--lr-min', default=0.01, type=float, help='min learning rate')
parser.add_argument('--lr-rampup', default=0, type=int, help='length of learning rate rampup in the beginning')
# parser.add_argument('--lr-rampdown-epochs', default=350, type=int,
#                     help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--interval', default=50, type=int, help='half interval of cosine')
parser.add_argument('--cycles', default=5, type=int, help='num of cycles of cyclic lr')
# parser.add_argument('--max-epoch', default=300, type=int, help='total epoch')

# (fast)SWA (swa type, epoch, ...)
# parser.add_argument('--max-total-epochs', default=600, type=int, help='total epochs to run')
parser.add_argument('--epochs', default=180, type=int, help='number of total epochs to run(notation \'l\' in paper)')
parser.add_argument('--num-cycles', default=5, type=int, help='additional cycles after args.epochs')
parser.add_argument('--cycle-interval', default=30, type=int, help='the number of epochs for small cyclic learning rate')
parser.add_argument('--cycle-rampdown-epochs', default=210, type=int, help='Half wavelength for the cosine annealing curve period(notation \'l_0\' in paper)')
# parser.add_argument('--max-epochs-per-filtering', default=300, type=int, help='max epochs for each filtering iteration')
parser.add_argument('--fastswa-frequencies', default='3', type=str, help='Average SWA every x epochs, even when on cycles')

# MT
parser.add_argument('--consistency', default=100.0, type=float,
                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency-type', default="mse", type=str, choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency-rampup', default=5, type=int, help='length of the consistency loss ramp-up')

# filtering
parser.add_argument('--exclude-unlabeled', default=False, type=bool,
                    help='exclude unlabeled examples from the training set')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--labeled-batch-size', default=62, type=int,
                    help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--filtering-type', default='snapshot_preds_ensemble', choices=['snapshot_preds_ensemble', 'swa_model', 'fastswa_model'], help='choose filtering type')

args = parser.parse_args()
args.name = "_".join([args.dataset, args.noise_type, str(args.noise_ratio), str(args.lr), str(args.ema_decay)])
