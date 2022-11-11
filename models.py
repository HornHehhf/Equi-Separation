'''
Pytorch models for CIFAR10: https://github.com/soapisnotfat/pytorch-cifar10
Pytorch models: https://github.com/pytorch/vision/tree/main/torchvision/models
'''

import torch.nn as nn
import torch.nn.functional as F

from utils import find_list_index


class GFNN(nn.Module):
    # feedforward neural networks
    def __init__(self, layer_num, input_size, hidden_size, output_size):
        super(GFNN, self).__init__()
        self.layer_num = layer_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inputs = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(inplace=True))
        layers = []
        for i in range(layer_num):
            layers += [nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.ReLU(inplace=True))]
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.size(1) == 3:
            x = x[:, 1, :, :]
        # print(x.shape)
        out = x.view(x.size(0), -1)
        out = self.inputs(out)
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out)
        out = self.classifier(out)
        return out

    def get_features(self, x, l):
        if x.size(1) == 3:
            x = x[:, 1, :, :]
        # print(x.shape)
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = self.inputs(out)
        if l == 1:
            return out
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out)
            if l == i + 2:
                return out


class FNNSW(nn.Module):
    # feedforward neural networks (Narrow-Wide)
    def __init__(self, input_size, hidden_size, output_size=10):
        super(FNNSW, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size * 10)
        self.bn5 = nn.BatchNorm1d(hidden_size * 10)
        self.fc6 = nn.Linear(hidden_size * 10, hidden_size * 10)
        self.bn6 = nn.BatchNorm1d(hidden_size * 10)
        self.fc7 = nn.Linear(hidden_size * 10, hidden_size * 10)
        self.bn7 = nn.BatchNorm1d(hidden_size * 10)
        self.fc8 = nn.Linear(hidden_size * 10, output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = F.relu(self.bn4(self.fc4(out)))
        out = F.relu(self.bn5(self.fc5(out)))
        out = F.relu(self.bn6(self.fc6(out)))
        out = F.relu(self.bn7(self.fc7(out)))
        out = self.fc8(out)
        return out

    def get_features(self, x, l):
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = F.relu(self.bn1(self.fc1(out)))
        if l == 1:
            return out
        out = F.relu(self.bn2(self.fc2(out)))
        if l == 2:
            return out
        out = F.relu(self.bn3(self.fc3(out)))
        if l == 3:
            return out
        out = F.relu(self.bn4(self.fc4(out)))
        if l == 4:
            return out
        out = F.relu(self.bn5(self.fc5(out)))
        if l == 5:
            return out
        out = F.relu(self.bn6(self.fc6(out)))
        if l == 6:
            return out
        out = F.relu(self.bn7(self.fc7(out)))
        if l == 7:
            return out


class FNNWS(nn.Module):
    # feedforward neural networks (Wide-Narrow)
    def __init__(self, input_size, hidden_size, output_size=10):
        super(FNNWS, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 10)
        self.bn1 = nn.BatchNorm1d(hidden_size * 10)
        self.fc2 = nn.Linear(hidden_size * 10, hidden_size * 10)
        self.bn2 = nn.BatchNorm1d(hidden_size * 10)
        self.fc3 = nn.Linear(hidden_size * 10, hidden_size * 10)
        self.bn3 = nn.BatchNorm1d(hidden_size * 10)
        self.fc4 = nn.Linear(hidden_size * 10, hidden_size * 10)
        self.bn4 = nn.BatchNorm1d(hidden_size * 10)
        self.fc5 = nn.Linear(hidden_size * 10, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)
        self.fc8 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = F.relu(self.bn4(self.fc4(out)))
        out = F.relu(self.bn5(self.fc5(out)))
        out = F.relu(self.bn6(self.fc6(out)))
        out = F.relu(self.bn7(self.fc7(out)))
        out = self.fc8(out)
        return out

    def get_features(self, x, l):
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = F.relu(self.bn1(self.fc1(out)))
        if l == 1:
            return out
        out = F.relu(self.bn2(self.fc2(out)))
        if l == 2:
            return out
        out = F.relu(self.bn3(self.fc3(out)))
        if l == 3:
            return out
        out = F.relu(self.bn4(self.fc4(out)))
        if l == 4:
            return out
        out = F.relu(self.bn5(self.fc5(out)))
        if l == 5:
            return out
        out = F.relu(self.bn6(self.fc6(out)))
        if l == 6:
            return out
        out = F.relu(self.bn7(self.fc7(out)))
        if l == 7:
            return out


class FNNMIX(nn.Module):
    # feedforward neural networks (Mixed)
    def __init__(self, input_size, hidden_size, output_size=10):
        super(FNNMIX, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 5)
        self.bn2 = nn.BatchNorm1d(hidden_size * 5)
        self.fc3 = nn.Linear(hidden_size * 5, hidden_size * 5)
        self.bn3 = nn.BatchNorm1d(hidden_size * 5)
        self.fc4 = nn.Linear(hidden_size * 5, hidden_size * 25)
        self.bn4 = nn.BatchNorm1d(hidden_size * 25)
        self.fc5 = nn.Linear(hidden_size * 25, hidden_size * 25)
        self.bn5 = nn.BatchNorm1d(hidden_size * 25)
        self.fc6 = nn.Linear(hidden_size * 25, hidden_size * 5)
        self.bn6 = nn.BatchNorm1d(hidden_size * 5)
        self.fc7 = nn.Linear(hidden_size * 5, hidden_size * 5)
        self.bn7 = nn.BatchNorm1d(hidden_size * 5)
        self.fc8 = nn.Linear(hidden_size * 5, output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = F.relu(self.bn4(self.fc4(out)))
        out = F.relu(self.bn5(self.fc5(out)))
        out = F.relu(self.bn6(self.fc6(out)))
        out = F.relu(self.bn7(self.fc7(out)))
        out = self.fc8(out)
        return out

    def get_features(self, x, l):
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = F.relu(self.bn1(self.fc1(out)))
        if l == 1:
            return out
        out = F.relu(self.bn2(self.fc2(out)))
        if l == 2:
            return out
        out = F.relu(self.bn3(self.fc3(out)))
        if l == 3:
            return out
        out = F.relu(self.bn4(self.fc4(out)))
        if l == 4:
            return out
        out = F.relu(self.bn5(self.fc5(out)))
        if l == 5:
            return out
        out = F.relu(self.bn6(self.fc6(out)))
        if l == 6:
            return out
        out = F.relu(self.bn7(self.fc7(out)))
        if l == 7:
            return out


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG11NoBN': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13NoBN': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16NoBN': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19NoBN': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

middle_num = {'VGG11': 8, 'VGG13': 10, 'VGG11NoBN': 8, 'VGG13NoBN': 10}


class VGG(nn.Module):
    def __init__(self, vgg_name, color_channel=3, num_classes=10):
        super(VGG, self).__init__()
        # original:
        # self.color_channel = color_channel
        # modified:
        self.color_channel = color_channel * 16
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.classifier = nn.Linear(4096, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.color_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # original:
                # layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                #            nn.BatchNorm2d(x),
                #            nn.ReLU(inplace=True)]
                # modified:
                layers += [nn.Conv2d(in_channels, x, kernel_size=5, padding=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.repeat(1, 16, 1, 1) # modified
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.classifier(out)
        return out

    def get_inputs(self, x):
        out = x.view(x.size(0), -1)
        return out

    def get_last_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        return out

    def get_penultimate_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        return out

    def get_third_from_last_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def get_features(self, x, l, vgg_name='VGG11'):
        x = x.repeat(1, 16, 1, 1)  # modified
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        elif 0 < l <= middle_num[vgg_name]:
            layer_config = cfg[vgg_name]
            layer_index_list = []
            layer_index = 0
            for i in range(len(layer_config)):
                if layer_config[i] == 'M':
                    layer_index += 1
                else:
                    layer_index += 3
                    layer_index_list.append(layer_index)
            cur_layer_index = layer_index_list[l - 1]
            modules = list(self.features.children())[:cur_layer_index]
            feature_model = nn.Sequential(*modules)
            out = feature_model(x)
            out = out.view(out.size(0), -1)
            return out
        else:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = F.relu(self.bn1(self.fc1(out)))
            if l == middle_num[vgg_name] + 1:
                return out
            out = F.relu(self.bn2(self.fc2(out)))
            if l == middle_num[vgg_name] + 2:
                return out


class AlexNet(nn.Module):
    def __init__(self, color_channel=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(color_channel, 64, kernel_size=11, stride=4, padding=5),
            nn.Conv2d(color_channel * 32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(192, 384, kernel_size=5, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # self.fc1 = nn.Linear(256, 4096)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.layer_index_list = [3, 7, 11, 14, 17]

    def forward(self, x):
        x = x.repeat(1, 32, 1, 1)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.classifier(out)
        return out

    def get_features(self, x, l):
        x = x.repeat(1, 32, 1, 1)
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        elif 0 < l <= 5:
            cur_layer_index = self.layer_index_list[l - 1]
            modules = list(self.features.children())[:cur_layer_index]
            feature_model = nn.Sequential(*modules)
            out = feature_model(x)
            out = out.view(out.size(0), -1)
            return out
        else:
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = F.relu(self.bn1(self.fc1(out)))
            if l == 6:
                return out
            out = F.relu(self.bn2(self.fc2(out)))
            if l == 7:
                return out


class BasicBlock(nn.Module):
    # two-layer block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # three-layer block
    # expansion = 4
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # original:
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # modified:
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # original:
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # modified:
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # original:
        # self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # modified:
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GResNet(nn.Module):
    # residual neural networks
    def __init__(self, block, num_blocks, color_channel=3, num_classes=10):
        super(GResNet, self).__init__()
        self.in_planes = 8

        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.classifier = nn.Linear(block.expansion*8*32*32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.repeat(1, 8, 1, 1)
        out = self.layer1(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_last_features(self, x):
        out = x.repeat(1, 8, 1, 1)
        out = self.layer1(out)
        out = out.view(out.size(0), -1)
        return out

    def get_basicblock_features(self, basicblock, index, x):
        conv1, bn1, conv2, bn2, shortcut = list(basicblock.children())
        out = F.relu(bn1(conv1(x)))
        if index == 0:
            return out
        out = bn2(conv2(out))
        out += shortcut(x)
        out = F.relu(out)
        return out

    def get_bottleneck_features(self, bottleneck, index, x):
        conv1, bn1, conv2, bn2, conv3, bn3, shortcut = list(bottleneck.children())
        out = F.relu(bn1(conv1(x)))
        if index == 0:
            return out
        out = F.relu(bn2(conv2(out)))
        if index == 1:
            return out
        out = bn3(conv3(out))
        out += shortcut(x)
        out = F.relu(out)
        return out

    def get_single_block_features(self, block, index, x, block_option):
        if block_option == 'basicblock':
            return self.get_basicblock_features(block, index, x)
        else:
            return self.get_bottleneck_features(block, index, x)

    def get_layer_features(self, x, l, block_option, num_blocks):
        if block_option == 'basicblock':
            block_size = 2
        else:
            block_size = 3
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        out = x.repeat(1, 8, 1, 1)
        if l == 1:
            out = out.view(out.size(0), -1)
            return out
        cur_part_layer = 2
        part_layer_index = [cur_part_layer]
        for i in range(len(num_blocks)):
            cur_part_layer += num_blocks[i] * block_size
            part_layer_index.append(cur_part_layer)
        part_index = find_list_index(part_layer_index, l)
        layer_list = [self.layer1]
        for i in range(part_index):
            out = layer_list[i](out)
        modules = list(layer_list[part_index].children())
        cur_block_layer = part_layer_index[part_index]
        block_layer_index = [cur_block_layer]
        for j in range(num_blocks[part_index]):
            cur_block_layer += block_size
            block_layer_index.append(cur_block_layer)
        block_index = find_list_index(block_layer_index, l)
        for j in range(block_index):
            out = self.get_single_block_features(modules[j], block_size - 1, out, block_option)
        out = self.get_single_block_features(modules[block_index], l - block_layer_index[block_index], out, block_option)
        out = out.view(out.size(0), -1)
        return out

    def get_block_features(self, x, l, block_option):
        if block_option == 'basicblock':
            block_size = 2
        else:
            block_size = 3
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        out = x.repeat(1, 8, 1, 1)
        if l == 1:
            out = out.view(out.size(0), -1)
            return out
        block_list = []
        block_list.extend(list(self.layer1.children()))
        for i in range(len(block_list)):
            basicblock = block_list[i]
            out = self.get_single_block_features(basicblock, block_size - 1, out, block_option)
            if l == i + 2:
                out = out.view(out.size(0), -1)
                return out

    def get_features(self, x, l, block_option='basicblock', num_blocks=[2], feature_option='block'):
        if feature_option == 'block':
            return self.get_block_features(x, l, block_option)
        else:
            return self.get_layer_features(x, l, block_option, num_blocks)


class ResNetMixV2(nn.Module):
    # residual neural networks with mixed blocks
    def __init__(self, num_blocks, color_channel=3, num_classes=10):
        super(ResNetMixV2, self).__init__()
        self.in_planes = 8
        assert num_blocks[0] % 2 == 0
        self.num_blocks = num_blocks[0]
        self.layer1 = self._make_layer(Bottleneck, 8, num_blocks[0] // 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 8, num_blocks[0] // 2, stride=1)
        self.classifier = nn.Linear(8*32*32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.repeat(1, 8, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_basicblock_features(self, basicblock, index, x):
        conv1, bn1, conv2, bn2, shortcut = list(basicblock.children())
        out = F.relu(bn1(conv1(x)))
        if index == 0:
            return out
        out = bn2(conv2(out))
        out += shortcut(x)
        out = F.relu(out)
        return out

    def get_bottleneck_features(self, bottleneck, index, x):
        conv1, bn1, conv2, bn2, conv3, bn3, shortcut = list(bottleneck.children())
        out = F.relu(bn1(conv1(x)))
        if index == 0:
            return out
        out = F.relu(bn2(conv2(out)))
        if index == 1:
            return out
        out = bn3(conv3(out))
        out += shortcut(x)
        out = F.relu(out)
        return out

    def get_single_block_features(self, block, index, x, block_option):
        if block_option == 'basicblock':
            return self.get_basicblock_features(block, index, x)
        else:
            return self.get_bottleneck_features(block, index, x)

    def get_block_features(self, x, l):
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        out = x.repeat(1, 8, 1, 1)
        if l == 1:
            out = out.view(out.size(0), -1)
            return out
        if 1 < l <= 1 + self.num_blocks // 2:
            block_list = []
            block_size = 3
            block_list.extend(list(self.layer1.children()))
            for i in range(len(block_list)):
                basicblock = block_list[i]
                out = self.get_single_block_features(basicblock, block_size - 1, out, block_option='bottleneck')
                if l == i + 2:
                    out = out.view(out.size(0), -1)
                    return out
        else:
            out = self.layer1(out)
            block_list = []
            block_size = 2
            block_list.extend(list(self.layer2.children()))
            for i in range(len(block_list)):
                basicblock = block_list[i]
                out = self.get_single_block_features(basicblock, block_size - 1, out, block_option='basicblock')
                if l == i + self.num_blocks // 2 + 2:
                    out = out.view(out.size(0), -1)
                    return out

    def get_layer_features(self, x, l):
        if l == 0:
            out = x.view(x.size(0), -1)
            return out
        out = x.repeat(1, 8, 1, 1)
        if l == 1:
            out = out.view(out.size(0), -1)
            return out
        if 1 < l <= 1 + (self.num_blocks // 2) * 3:
            block_option = 'bottleneck'
            block_size = 3
            modules = list(self.layer1.children())
            cur_block_layer = 2
            block_layer_index = [cur_block_layer]
            for j in range(self.num_blocks // 2):
                cur_block_layer += block_size
                block_layer_index.append(cur_block_layer)
            block_index = find_list_index(block_layer_index, l)
            for j in range(block_index):
                out = self.get_single_block_features(modules[j], block_size - 1, out, block_option)
            out = self.get_single_block_features(modules[block_index], l - block_layer_index[block_index], out, block_option)
            out = out.view(out.size(0), -1)
        else:
            out = self.layer1(out)
            block_option = 'basicblock'
            block_size = 2
            modules = list(self.layer2.children())
            cur_block_layer = 2 + (self.num_blocks // 2) * 3
            block_layer_index = [cur_block_layer]
            for j in range(self.num_blocks // 2):
                cur_block_layer += block_size
                block_layer_index.append(cur_block_layer)
            block_index = find_list_index(block_layer_index, l)
            for j in range(block_index):
                out = self.get_single_block_features(modules[j], block_size - 1, out, block_option)
            out = self.get_single_block_features(modules[block_index], l - block_layer_index[block_index], out,
                                                 block_option)
            out = out.view(out.size(0), -1)
        return out

    def get_features(self, x, l, feature_option='block'):
        if feature_option == 'layer':
            return self.get_layer_features(x, l)
        else:
            return self.get_block_features(x, l)

class GFNNOriginal(nn.Module):
    # Feedforward neural networks for original images: Fashion-MNIST (32, 32) and CIFAR-10 (3, 32, 32)
    def __init__(self, layer_num, input_size, hidden_size, output_size):
        super(GFNNOriginal, self).__init__()
        self.layer_num = layer_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inputs = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(inplace=True))
        layers = []
        for i in range(layer_num):
            layers += [nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.BatchNorm1d(hidden_size),
                                     nn.ReLU(inplace=True))]
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # print(x.shape)
        out = x.view(x.size(0), -1)
        out = self.inputs(out)
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out)
        out = self.classifier(out)
        return out

    def get_features(self, x, l):
        # print(x.shape)
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = self.inputs(out)
        if l == 1:
            return out
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out)
            if l == i + 2:
                return out

