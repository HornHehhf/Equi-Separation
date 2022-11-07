#!/usr/bin/env bash
if [[ $4 = adam ]]
then
    for lr in 3e-5 1e-4 3e-4 1e-3 3e-3
    do
        echo "run setting: model $1 block num $2 feature option $3 optimization $4 lr $lr"
        python representation_dynamics_terminal_mixresidual.py data=fashion_mnist model=$1 block_option=MIX block_num=$2 feature_option=$3 measure=within_variance optimization=$4 lr=$lr
    done
else
    for lr in 0.001 0.003 0.01 0.03 0.1 0.3 1.0
    do
        echo "run setting: model $1 block num $2 feature option $3 optimization $4 lr $lr"
        python representation_dynamics_terminal_mixresidual.py data=fashion_mnist model=$1 block_option=MIX block_num=$2 feature_option=$3 measure=within_variance optimization=$4 lr=$lr
    done
fi

