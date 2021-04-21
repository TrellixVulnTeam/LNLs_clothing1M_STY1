import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--num-classes', default=None, type=int, help='number of classes')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'clothes1M', 'webvision'], help='choose dataset')
parser.add_argument('--val-size', default=5000, type=int, help='size of validation set')
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
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--final-run', action='store_true', help='train without validation set')


# noise
parser.add_argument('--noise-type', default='symmetric', choices=['symmetric', 'asymmetric'], help='type of noise')
parser.add_argument('--noise-ratio', default=0.4, type=float, help='ratio of noise')
parser.add_argument('--noisy-validation', default=True, type=bool, help='clean validation or noisy validation set')

# lr scheduling
parser.add_argument('--lr-type', default='cosineannealing', type=str, choices=['fastswa', 'cosineannealing'], help='learning rate schedule type')
parser.add_argument('--lr', default=0.05, type=float, help='max learning rate')
parser.add_argument('--lr-min', default=0.0, type=float, help='min learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float, help='initial learning rate when using linear rampup')

# lr cosineannealing (준호 참고)
parser.add_argument('--interval', default=10, type=int, help='interval of a cycle')
parser.add_argument('--cycles', default=5, type=int, help='num of cycles')

# (fast)SWA (swa type, epoch, ...) (재순이형 참고)
parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
parser.add_argument('--first-interval', default=180, type=int, help='number of total epochs to run(notation \'l\' in paper)')
parser.add_argument('--cycle-rampdown-epochs', default=210, type=int, help='Half wavelength for the cosine annealing curve period(notation \'l_0\' in paper)')
parser.add_argument('--interval', default=30, type=int, help='the number of epochs for small cyclical learning rate')
parser.add_argument('--cycles', default=5, type=int, help='additional cycles after args.epochs')
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
