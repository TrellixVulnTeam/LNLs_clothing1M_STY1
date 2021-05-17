

for i in 0.002 0.004 0.006 0.008 0.01 0.001 0.0008
do
    python3 warmup_analyze_clothing1M.py\
        --train-size 64000\
        --dataset clothing1M\
        --decay 1e-3\
        --lr-type 'cosineannealing'\
        --lr $i\
        --lr-min 0.0001\
        --interval 10\
        --cycles 10\
        --batch-size 32\
        --arch resnet50     #resnet_clothing1M에서 불러옴
    
done
