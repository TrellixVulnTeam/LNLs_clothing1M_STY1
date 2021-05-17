import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--num-classes', default=None, type=int, help='number of classes')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'clothing1M', 'webvision'], help='choose dataset')
parser.add_argument('--val-size', default=5000, type=int, help='size of validation set')
parser.add_argument('--train-size', default=64000, type=int, help='size of validation set')
parser.add_argument('--arch', default="resnet50", type=str, help='model architecture')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='use nesterov momentum')
parser.add_argument('--ema-decay', default=0.97, type=float, help='ema variable decay rate (default: 0.99)')
parser.add_argument('--logit-distance-cost', default=.01, type=float,
                    help='let the student model have two outputs and use an MSE loss between the logits with\
                     the given weight (default: only have one output)')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--final-run', action='store_true', help='train without validation set')

parser.add_argument('--interval', default=10, type=int, help='interval of a cycle')
parser.add_argument('--cycles', default=5, type=int, help='num of cycles')

# noise
parser.add_argument('--noise-type', default='symmetric', choices=['symmetric', 'asymmetric'], help='type of noise')
parser.add_argument('--noise-ratio', default=0.4, type=float, help='ratio of noise')
parser.add_argument('--noisy-validation', default=True, type=bool, help='clean validation or noisy validation set')

# (fast)SWA (swa type, epoch, ...)
parser.add_argument('--lr', default=0.002, type=float, help='max learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float, help='initial learning rate when using linear rampup')



parser.add_argument('--SSL', action='store_true', help='start ssl with warmup pickle file')

# parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
parser.add_argument('--epochs', default=180, type=int, help='number of total epochs to run(notation \'l\' in paper)')
parser.add_argument('--cycle-rampdown-epochs', default=210, type=int, help='Half wavelength for the cosine annealing curve period(notation \'l_0\' in paper)')
parser.add_argument('--num-cycles', default=5, type=int, help='additional cycles after args.epochs')
parser.add_argument('--cycle-interval', default=30, type=int, help='the number of epochs for small cyclical learning rate')
parser.add_argument('--fastswa-frequencies', default='3', type=str, help='Average SWA every x epochs, even when on cycles')

# WARMUP
parser.add_argument('--lr-max-warmup', default=0.05, type=float, help='start lr value (for warmup scheduler)')
parser.add_argument('--lr-min-warmup', default=0.001, type=float, help='min lr value (for warmup scheduler)')
parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
parser.add_argument('--warmup-epoch', default=100, type=int, help='total warming up epoch (using fully labeled trainset)')
parser.add_argument('--lr-scheduler-warmup', default="cosineannealingwarmrestarts", type=str, choices=['exponential', 'cosineanealing', 'step', 'cosineannealingwarmrestarts'],
                    help='lr scheduling type')
parser.add_argument('--lr-cycles-warmup', default=10, type=int, help='lr cycles (for warmup scheduler)')
parser.add_argument('--lr-interval-warmup', default=10, type=int, help='interval of lr cycle (for warmup scheduler)')
parser.add_argument('--batch-size-warmup', default=64, type=int, help='batch size for warmup')

# FUMM
parser.add_argument('--lr-max', default=0.002, type=float, help='start lr value (for all scheduler)') #
parser.add_argument('--lr-min', default=0.0001, type=float, help='min learning rate') #
parser.add_argument('--max-total-epoch', default=600, type=int, help='max total epoch')

parser.add_argument('--lr-scheduler', default="cosineannealingwarmrestarts", type=str, choices=['exponential', 'cosineanealing', 'step', 'cosineannealingwarmrestarts'],
                    help='lr scheduling type') #x

parser.add_argument('--lr-cycles', default=10, type=int, help='lr cycles')
parser.add_argument('--lr-interval', default=10, type=int, help='interval of lr cycle')
parser.add_argument('--lr-interval-ssl', default=10, type=int, help='interval of lr cycle during ssl')
parser.add_argument('--lr-stepsize', default=10, type=float, help='used for stepLR')
parser.add_argument('--model-state', default='initialize', type=str, choices=['initialize', 'continue', 'load_best'],
                    help='model state when starting ssl stage')
parser.add_argument('--if-ssl-cyclic', default=False, type=bool,
                    help='how to set lr scheduler during ssl stage, cyclic or non-cyclic')
parser.add_argument('--resume', default=True, type=bool,
                    help='start with')


# MT
parser.add_argument('--consistency', default=100.0, type=float,
                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency-type', default="mse", type=str, choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency-rampup', default=5, type=int, help='length of the consistency loss ramp-up')


# filtering
parser.add_argument('--exclude-unlabeled', default=False, type=bool,
                    help='exclude unlabeled examples from the training set')
parser.add_argument('--batch-size', default=64, type=int, help='batch size') #256
parser.add_argument('--labeled-batch-size', default=16, type=int,    #62
                    help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--filtering-type', default='snapshot_preds_ensemble', choices=['snapshot_preds_ensemble', 'swa_model', 'fastswa_model'], help='choose filtering type')

args = parser.parse_args()
args.name = "_".join([args.dataset, args.noise_type, str(args.noise_ratio), str(args.lr), str(args.ema_decay)])