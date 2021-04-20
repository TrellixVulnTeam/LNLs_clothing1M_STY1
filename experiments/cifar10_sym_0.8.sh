python3 train_self.py\
    --dataset cifar10\
    --noise-type symmetric\
    --noise-ratio 0.8\
    --batch-size 256\
    --labeled-batch-size 62\
    --lr 0.05\
    --decay 2e-4\
    --ema-decay 0.99\
    --val-size 5000\
    --max-total-epochs 600\
    --max-epochs-per-filtering 300\
    --patience 50
    
