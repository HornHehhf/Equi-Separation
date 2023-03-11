# optimization
## SGD
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=sgd lr=0.3 > logs/cfm_GFNN_2_100_sgd_within_0.3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=sgd lr=0.03 > logs/cfm_GFNN_6_100_sgd_within_0.03_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=sgd lr=0.01 > logs/cfm_GFNN_18_100_sgd_within_0.01_terminal.log 2>&1 &
## SGD with momentum
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=momentum lr=0.3 > logs/cfm_GFNN_2_100_momentum_within_0.3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=momentum lr=0.01 > logs/cfm_GFNN_6_100_momentum_within_0.01_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=momentum lr=0.001 > logs/cfm_GFNN_18_100_momentum_within_0.001_terminal.log 2>&1 &
## Adam
CUDA_VISIBLE_DEVICES=5 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=2 hidden_size=100 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_GFNN_2_100_adam_within_3e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_GFNN_6_100_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=18 hidden_size=100 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_GFNN_18_100_adam_within_3e-4_terminal.log 2>&1 &

# learning rate
CUDA_VISIBLE_DEVICES=7 nohup sh run_optimization.sh cfm 7 sgd > logs/cfm_GFNN_7_100_sgd_within_terminal.log 2>&1 &

# Imbalanced data
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_imbalanced.sh 2 adam > logs/cfm_GFNN_2_100_adam_within_imbalanced_terminal_new.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_imbalanced.sh 6 adam > logs/cfm_GFNN_6_100_adam_within_imbalanced_terminal_new.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization_imbalanced.sh 18 adam > logs/cfm_GFNN_18_100_adam_within_imbalanced_terminal_new.log 2>&1 &

# Depth
## MNIST
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cm 1 adam > logs/cm_GFNN_1_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization.sh cm 2 adam > logs/cm_GFNN_2_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization.sh cm 3 adam > logs/cm_GFNN_3_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_optimization.sh cm 4 adam > logs/cm_GFNN_4_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_optimization.sh cm 5 adam > logs/cm_GFNN_5_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_optimization.sh cm 6 adam > logs/cm_GFNN_6_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_optimization.sh cm 7 adam > logs/cm_GFNN_7_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_optimization.sh cm 8 adam > logs/cm_GFNN_8_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cm 16 adam > logs/cm_GFNN_16_100_adam_within_terminal.log 2>&1 &
## Fashion-MNIST
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cfm 1 adam > logs/cfm_GFNN_1_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization.sh cfm 3 adam > logs/cfm_GFNN_3_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization.sh cfm 4 adam > logs/cfm_GFNN_4_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_optimization.sh cfm 5 adam > logs/cfm_GFNN_5_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_optimization.sh cfm 6 adam > logs/cfm_GFNN_6_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_optimization.sh cfm 7 adam > logs/cfm_GFNN_7_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_optimization.sh cfm 8 adam > logs/cfm_GFNN_8_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_optimization.sh cfm 9 adam > logs/cfm_GFNN_9_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cfm 16 adam > logs/cfm_GFNN_16_100_adam_within_terminal.log 2>&1 &
## CIFAR-10
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cc 1 adam > logs/cc_GFNN_1_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization.sh cc 2 adam > logs/cc_GFNN_2_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization.sh cc 6 adam > logs/cc_GFNN_6_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_optimization.sh cc 7 adam > logs/cc_GFNN_7_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_optimization.sh cc 8 adam > logs/cc_GFNN_8_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_optimization.sh cc 9 adam > logs/cc_GFNN_9_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_optimization.sh cc 10 adam > logs/cc_GFNN_10_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_optimization.sh cc 11 adam > logs/cc_GFNN_11_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization.sh cc 12 adam > logs/cc_GFNN_12_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization.sh cc 16 adam > logs/cc_GFNN_16_100_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization.sh cc 18 adam > logs/cc_GFNN_18_100_adam_within_terminal.log 2>&1 &

# Width
## Width = 20
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=20 measure=within_variance optimization=adam lr=3e-5 > logs/cfm_GFNN_6_20_adam_within_3e-5_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=20 measure=within_variance optimization=adam lr=1e-4 > logs/cfm_GFNN_6_20_adam_within_1e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=20 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_GFNN_6_20_adam_within_3e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=20 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_GFNN_6_20_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=20 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_GFNN_6_20_adam_within_3e-3_terminal.log 2>&1 &
## Width=1000
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=1000 measure=within_variance optimization=adam lr=3e-5 > logs/cfm_GFNN_6_1000_adam_within_3e-5_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=1000 measure=within_variance optimization=adam lr=1e-4 > logs/cfm_GFNN_6_1000_adam_within_1e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=1000 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_GFNN_6_1000_adam_within_3e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=1000 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_GFNN_6_1000_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal.py data=cfm model=GFNN layer_num=6 hidden_size=1000 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_GFNN_6_1000_adam_within_3e-3_terminal.log 2>&1 &

# shape
## Narrow-Wide
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNSW layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-5 > logs/cfm_FNNSW_6_100_adam_within_3e-5_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNSW layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-4 > logs/cfm_FNNSW_6_100_adam_within_1e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNSW layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_FNNSW_6_100_adam_within_3e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNSW layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_FNNSW_6_100_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNSW layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_FNNSW_6_100_adam_within_3e-3_terminal.log 2>&1 &
## Wide-Narrow
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNWS layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-5 > logs/cfm_FNNWS_6_100_adam_within_3e-5_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNWS layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-4 > logs/cfm_FNNWS_6_100_adam_within_1e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNWS layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_FNNWS_6_100_adam_within_3e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNWS layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_FNNWS_6_100_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNWS layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_FNNWS_6_100_adam_within_3e-3_terminal.log 2>&1 &
## Mixed
CUDA_VISIBLE_DEVICES=0 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNMIX layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-5 > logs/cfm_FNNMIX_6_100_adam_within_3e-5_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNMIX layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-4 > logs/cfm_FNNMIX_6_100_adam_within_1e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNMIX layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-4 > logs/cfm_FNNMIX_6_100_adam_within_3e-4_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNMIX layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=1e-3 > logs/cfm_FNNMIX_6_100_adam_within_1e-3_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python representation_dynamics_terminal_shape.py data=cfm model=FNNMIX layer_num=6 hidden_size=100 measure=within_variance optimization=adam lr=3e-3 > logs/cfm_FNNMIX_6_100_adam_within_3e-3_terminal.log 2>&1 &

# Convolutional neural networks
## VGG
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_CNN.sh fashion_mnist VGG13 11 adam > logs/fashion_mnist_VGG13_11_adam_within_termnial_kernel5_channel16.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_CNN.sh cifar10 VGG13 11 adam > logs/cifar10_VGG13_11_adam_within_termnial_kernel5_channel16.log 2>&1 &
## AlexNet
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_CNN.sh fashion_mnist AlexNet 6 adam > logs/fashion_mnist_AlexNet_adam_within_termnial.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_CNN.sh cifar10 AlexNet 6 adam > logs/cifar10_AlexNet_adam_within_termnial.log 2>&1 &

# Residual neural networks
## 2-layer block
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_residual.sh basicblock 4 block adam > logs/fashion_mnist_GResNet_basicblock_4_block_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_residual.sh basicblock 4 layer adam > logs/fashion_mnist_GResNet_basicblock_4_layer_adam_within_terminal.log 2>&1 &
## 3-layer block
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization_residual.sh bottleneck 4 block adam > logs/fashion_mnist_GResNet_bottleneck_4_block_adam_within_terminal_kernel5.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_optimization_residual.sh bottleneck 4 layer adam > logs/fashion_mnist_GResNet_bottleneck_4_layer_adam_within_terminal_kernel5.log 2>&1 &
## Mixed
CUDA_VISIBLE_DEVICES=4 nohup sh run_optimization_mixresidual.sh ResNetMixV2 4 block adam > logs/fashion_mnist_ResNetMixV2_4_block_adam_within_terminal_kernel5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_optimization_mixresidual.sh ResNetMixV2 4 layer adam > logs/fashion_mnist_ResNetMixV2_4_layer_adam_within_terminal_kernel5.log 2>&1 &

# Frozen neural networks
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_frozen.sh cc 18 adam > logs/cc_GFNN_18_100_adam_within_terminal_frozen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_frozen.sh cc 18 adam > logs/cc_GFNN_18_100_adam_within_terminal_unfrozen.log 2>&1 &

# Feedforward neural networks on original images: Fashion-MNIST (32, 32) and CIFAR10 (3, 32, 32)
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_original.sh fashion_mnist 2 1000 adam > logs/paper/fashion_mnist_GFNNOriginal_2_1000_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_optimization_original.sh fashion_mnist 6 1000 adam > logs/paper/fashion_mnist_GFNNOriginal_6_1000_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_optimization_original.sh fashion_mnist 18 1000 adam > logs/paper/fashion_mnist_GFNNOriginal_18_1000_adam_within_terminal.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup sh run_optimization_original.sh cifar10 2 1000 adam > logs/paper/cifar10_GFNNOriginal_2_1000_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_optimization_original.sh cifar10 6 1000 adam > logs/paper/cifar10_GFNNOriginal_6_1000_adam_within_terminal.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_optimization_original.sh cifar10 18 1000 adam > logs/paper/cifar10_GFNNOriginal_18_1000_adam_within_terminal.log 2>&1 &

# Linearized neural networks
CUDA_VISIBLE_DEVICES=0 nohup sh run_optimization_linearized.sh 18 adam > logs/paper/cfm_GLNN_18_100_adam_within_terminal.log 2>&1 &

# No Batch normalization
CUDA_VISIBLE_DEVICES=7 nohup sh run_optimization_NoBN.sh 18 adam > logs/paper/cfm_GFNNNoBN_18_100_adam_within_terminal.log 2>&1 &
