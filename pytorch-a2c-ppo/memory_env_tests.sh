#!/usr/bin/env bash

policies=("ff" 'rnn' 'tc' 'mha' 'cached_tc' 'cached_mha')

for pol in "${policies[@]}"; do
    #for per in "${update_period[@]}"; do
    echo "##################################################"
    echo "Train ppo-${pol} with episode-len=50"

    python3 train_memory_env.py \
        -pol $pol -fs 50 -el 49 --num-steps 50 -hs 64 --num-processes 32 -et "test/eplen50-${pol}" --entropy-coef 0.01 -sd pretrained/mem-tests-50 --algo ppo --use-gae --lr 3e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-mini-batch 4 --log-interval 10 --num-env-steps 1000000 --use-linear-lr-decay

done
