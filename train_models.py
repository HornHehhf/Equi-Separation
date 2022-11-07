import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import get_minibatches_idx
from models import GFNN, FNNSW, FNNWS, FNNMIX, VGG, AlexNet, GResNet, BasicBlock, Bottleneck, ResNetMixV2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def build_model(config):
    if config['model'] == 'GFNN':
        model = GFNN(config['layer_num'], config['input_size'], config['hidden_size'], config['class_num'])
    elif config['model'] == 'FNNWS':
        model = FNNWS(config['input_size'], config['hidden_size'], output_size=config['class_num'])
    elif config['model'] == 'FNNSW':
        model = FNNSW(config['input_size'], config['hidden_size'], output_size=config['class_num'])
    elif config['model'] == 'FNNMIX':
        model = FNNMIX(config['input_size'], config['hidden_size'], output_size=config['class_num'])
    elif config['model'] == 'VGG13':
        model = VGG('VGG13', color_channel=config['color_channel'])
    elif config['model'] == 'AlexNet':
        model = AlexNet(color_channel=config['color_channel'])
    elif config['model'] == 'GResNet':
        if config['block_option'] == 'basicblock':
            model = GResNet(BasicBlock, config['num_blocks'], color_channel=config['color_channel'],
                            num_classes=config['class_num'])
        else:
            model = GResNet(Bottleneck, config['num_blocks'], color_channel=config['color_channel'],
                            num_classes=config['class_num'])
    elif config['model'] == 'ResNetMixV2':
        model = ResNetMixV2(config['num_blocks'], color_channel=config['color_channel'], num_classes=config['class_num'])
    else:
        print('wrong model option')
        model = None
    loss_function = nn.CrossEntropyLoss()
    if config['optimization'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimization'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],  momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    return model, loss_function, optimizer


def simple_test_batch(testloader, model, config):
    model.eval()
    total = 0.0
    correct = 0.0
    print('data size', len(testloader))
    minibatches_idx = get_minibatches_idx(len(testloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(testloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(testloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.long()).sum().item()
    test_accuracy = correct / total
    return test_accuracy

