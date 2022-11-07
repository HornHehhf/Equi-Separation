#!/usr/bin/env bash
if [[ $2 = adam ]]
then
    for lr in 3e-5 1e-4 3e-4 1e-3 3e-3
    do
        echo "run setting: layer number $1 optimization $2 lr $lr"
        python representation_dynamics_terminal_imbalance.py data=cfm model=GFNN layer_num=$1 hidden_size=100 measure=within_variance optimization=$2 lr=$lr
    done
else
    for lr in 0.001 0.003 0.01 0.03 0.1 0.3 1.0
    do
        echo "run setting: layer number $1 optimization $2 lr $lr"
        python representation_dynamics_terminal_imbalance.py data=cfm model=GFNN layer_num=$1 hidden_size=100 measure=within_variance optimization=$2 lr=$lr
    done
fi