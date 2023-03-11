# Equi-Separation Law
This is the code repository for the ArXiv paper [A Law of Data Separation in Deep Learning](https://arxiv.org/pdf/2210.17020.pdf).
If you use this code for your work, please cite
```
@article{he2022law,
  title={A Law of Data Separation in Deep Learning},
  author={He, Hangfeng and Su, Weijie J},
  journal={arXiv preprint arXiv:2210.17020},
  year={2022}
}
```
## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python==3.7.10\
pip3 install -r requirements.txt

## Code Organization
- Utilities (utils.py)
- Data preprocessing (data.py, imbalanced_data.py)
- Model utilities (models.py, train_models.py)
- Representations (variance_analysis.py, representation_analysis.py)
- Terminal analysis (representation_dynamics_terminal.py, representation_dynamics_terminal_*.py)
- Optimization scripts (run_optimization.sh, run_optimization_*.sh)
- Experiments scripts (run_experiments.sh)

## Change the Dir Path
Change the /path/to/working/dir to your working directory

## Reproducing experiments
You need to save the data before you run the experiments (the config in data.py and imbalanced_data.py can be changed for your purpose)
```
python data.py data=cm
python data.py data=cfm
python data.py data=cc
python data.py data=fashion_mnist
python data.py data=cifar10
python imbalanced_data.py data=cfm
```

To reproduce the main experiments
```
sh run_experiments.sh
```
Note that there are too many experiments in run_experiments.sh, so please run it by keeping some part and commenting the remaining part

You can also choose to reproduce specific experiments, e.g., Figure 1 in the paper
```
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=sgd lr=0.3 > logs/cfm_GFNN_2_100_sgd_within_0.3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=sgd lr=0.03 > logs/cfm_GFNN_6_100_sgd_within_0.03_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=sgd lr=0.01 > logs/cfm_GFNN_18_100_sgd_within_0.01_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=momentum lr=0.3 > logs/cfm_GFNN_2_100_momentum_within_0.3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=momentum lr=0.01 > logs/cfm_GFNN_6_100_momentum_within_0.01_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=momentum lr=0.001 > logs/cfm_GFNN_18_100_momentum_within_0.001_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_GFNN_2_100_adam_within_3e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_GFNN_6_100_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_GFNN_18_100_adam_within_3e-4_terminal.log 2>&1 &
```

## More experiments
Equi-separation law also exists in feedforward neural networks (width=1000) with different depths on original images: Fashion-MNIST (32, 32) and CIFAR-10 (3, 32, 32)
- Fashion-MNIST (figures/OriginalSize/fashion_mnist*png)
- CIFAR-10 (figures/OriginalSize/cifar10*png)

Equi-separation law does not exist in feedforward neural networks without batch normalization: figures/NoBN/*png

Equi-separation law does not exist in linearized neural networks (without nonlinear activation function): figures/Linearized/*png