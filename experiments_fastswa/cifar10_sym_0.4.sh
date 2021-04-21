python3 main_.py\
    --dataset cifar10\
    --noise-type symmetric\
    --noise-ratio 0.4\
    --batch-size 128\
    --labeled-batch-size 31\
    --lr 0.05\
    --lr-min 0.0\
    --decay 2e-4\
    --ema-decay 0.97\
    --val-size 5000\
    --epochs 180\
    --cycle-rampdown-epochs 210\
    --num-cycles 5\
    --cycle-interval 30\
    --fastswa-frequencies '3'
