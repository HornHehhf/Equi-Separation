import torch
import numpy as np
from torch.autograd import Variable
from sklearn.decomposition import PCA


from utils import get_minibatches_idx
from variance_analysis import get_variation, get_variation_imbalance

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_features_labels(trainloader, model, config, option=0):
    total_features = []
    total_labels = []
    minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        if config['model'] == 'GResNet':
            features = model.get_features(inputs, option, block_option=config['block_option'],
                                          num_blocks=config['num_blocks'], feature_option=config['feature_option'])
        elif config['model'] == 'ResNetMixV2':
            features = model.get_features(inputs, option, feature_option=config['feature_option'])
        elif config['model'][: 3] == 'VGG':
            features = model.get_features(inputs, option, vgg_name=config['model'])
        else:
            features = model.get_features(inputs, option)
        total_features.extend(features.cpu().data.numpy().tolist())
        total_labels.extend(targets.cpu().data.numpy().tolist())
    total_features = np.array(total_features)
    total_labels = np.array(total_labels).astype(int)
    return total_features, total_labels


def analyze_representations(train_data, model, config):
    rate_reduction_list = []
    for feature_option in range(config['layer_num'] + 2):
        total_features, total_labels = get_features_labels(train_data, model, config, option=feature_option)
        if config['measure'] == 'within_variance':
            rate_reduction = get_variation(total_features, total_labels, config)[0]
        else:
            rate_reduction = get_variation(total_features, total_labels, config)[1]
        print('{}-layer {}'.format(feature_option, config['measure']), rate_reduction)
        rate_reduction_list.append(rate_reduction)
    return rate_reduction_list


def analyze_representations_residual(train_data, model, config):
    rate_reduction_list = []
    for feature_option in range(config['layer_num'] + 2):
        total_features, total_labels = get_features_labels(train_data, model, config, option=feature_option)
        if total_features.shape[-1] > 100:
            print('feature dim', total_features.shape[-1])
            pca = PCA(n_components=100, svd_solver='full')
            total_features = pca.fit_transform(total_features)
            print('reduced feature dim', total_features.shape[-1])
        if config['measure'] == 'within_variance':
            rate_reduction = get_variation(total_features, total_labels, config)[0]
        else:
            rate_reduction = get_variation(total_features, total_labels, config)[1]
        print('{}-layer {}'.format(feature_option, config['measure']), rate_reduction)
        rate_reduction_list.append(rate_reduction)
    return rate_reduction_list


def analyze_representations_imbalance(train_data, model, config):
    rate_reduction_list = []
    for feature_option in range(config['layer_num'] + 2):
        total_features, total_labels = get_features_labels(train_data, model, config, option=feature_option)
        if config['measure'] == 'within_variance':
            rate_reduction = get_variation_imbalance(total_features, total_labels, config)[0]
        else:
            rate_reduction = get_variation_imbalance(total_features, total_labels, config)[1]
        print('{}-layer {}'.format(feature_option, config['measure']), rate_reduction)
        rate_reduction_list.append(rate_reduction)
    return rate_reduction_list
