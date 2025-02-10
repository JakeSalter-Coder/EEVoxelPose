from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, server=True, residual=0, subpartition=0):
        if server:
            if subpartition == 0:
                residual = x

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.conv3(x)
                x = self.bn3(x)

                if self.downsample is not None:
                    residual = self.downsample(residual)

                x += residual
                x = self.relu(x)
            elif subpartition == 1:
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.conv3(x)
                x = self.bn3(x)

                if self.downsample is not None:
                    residual = self.downsample(residual)

                x += residual
                x = self.relu(x)
            elif subpartition == 2:
                x = self.conv3(x)
                x = self.bn3(x)

                if self.downsample is not None:
                    residual = self.downsample(residual)

                x += residual
                x = self.relu(x)
            elif subpartition == 3:
                if self.downsample is not None:
                    residual = self.downsample(residual)

                x += residual
                x = self.relu(x)
            elif subpartition == 4:
                x = x
            return x
        else:
            if subpartition == 0:
                x = x
            elif subpartition == 1:
                residual = x

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
            elif subpartition == 2:
                residual = x

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
            elif subpartition == 3:
                residual = x

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.conv3(x)
                x = self.bn3(x)
            elif subpartition == 4:
                residual = x

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.conv3(x)
                x = self.bn3(x)

                if self.downsample is not None:
                    residual = self.downsample(residual)

                x += residual
                x = self.relu(x)
            return x, residual


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg):
        self.inplanes = 64
        self.deconv_with_bias = cfg.RESNET.DECONV_WITH_BIAS

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.RESNET.NUM_DECONV_LAYERS,
            cfg.RESNET.NUM_DECONV_FILTERS,
            cfg.RESNET.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.RESNET.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.DATASET.NUM_JOINTS,
            kernel_size=cfg.RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.RESNET.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, residual=0, server=True, partition=0):
        if server:
            if partition == 0:
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 1:
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer1, block1
            elif partition == 2:
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 3:
                x = self.layer1[0](x, residua=residual, subpartition=1)
                x = self.layer1[1:](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 4:
                x = self.layer1[0](x, residua=residual, subpartition=2)
                x = self.layer1[1:](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 5:
                x = self.layer1[0](x, residua=residual, subpartition=3)
                x = self.layer1[1:](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer1, block2
            elif partition == 6:
                x = self.layer1[1:](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 7:
                x = self.layer1[1](x, residua=residual, subpartition=1)
                x = self.layer1[2](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 8:
                x = self.layer1[1](x, residua=residual, subpartition=2)
                x = self.layer1[2](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 9:
                x = self.layer1[1](x, residua=residual, subpartition=3)
                x = self.layer1[2](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer1, block3
            elif partition == 10:
                x = self.layer1[2](x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 11:
                x = self.layer1[2](x, residua=residual, subpartition=1)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 12:
                x = self.layer1[2](x, residua=residual, subpartition=2)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 13:
                x = self.layer1[2](x, residua=residual, subpartition=3)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer2, block 1
            elif partition == 14:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 15:
                x = self.layer2[0](x, residua=residual, subpartition=1)
                x = self.layer2[1:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 16:
                x = self.layer2[0](x, residua=residual, subpartition=2)
                x = self.layer2[1:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 17:
                x = self.layer2[0](x, residua=residual, subpartition=3)
                x = self.layer2[1:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            #partition layer2, block2
            elif partition == 18:
                x = self.layer2[1:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 19:
                x = self.layer2[1](x, residua=residual, subpartition=1)
                x = self.layer2[2:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 20:
                x = self.layer2[1](x, residua=residual, subpartition=2)
                x = self.layer2[2:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 21:
                x = self.layer2[1](x, residua=residual, subpartition=3)
                x = self.layer2[2:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer2, block3
            elif partition == 22:
                x = self.layer2[2:](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 23:
                x = self.layer2[2](x, residua=residual, subpartition=1)
                x = self.layer2[3](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 24:
                x = self.layer2[2](x, residua=residual, subpartition=2)
                x = self.layer2[3](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 25:
                x = self.layer2[2](x, residua=residual, subpartition=3)
                x = self.layer2[3](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer2, block4
            elif partition == 25:
                x = self.layer2[3](x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 25:
                x = self.layer2[3](x, residua=residual, subpartition=1)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 26:
                x = self.layer2[3](x, residua=residual, subpartition=2)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 27:
                x = self.layer2[3](x, residua=residual, subpartition=3)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block1
            elif partition == 27:
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 28:
                x = self.layer3[0](x, residua=residual, subpartition=1)
                x = self.layer3[1:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 29:
                x = self.layer3[0](x, residua=residual, subpartition=2)
                x = self.layer3[1:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 30:
                x = self.layer3[0](x, residua=residual, subpartition=3)
                x = self.layer3[1:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block2
            elif partition == 31:
                x = self.layer3[1:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 32:
                x = self.layer3[1](x, residua=residual, subpartition=1)
                x = self.layer3[2:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 33:
                x = self.layer3[1](x, residua=residual, subpartition=2)
                x = self.layer3[2:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 34:
                x = self.layer3[1](x, residua=residual, subpartition=3)
                x = self.layer3[2:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block3
            elif partition == 35:
                x = self.layer3[2](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 36:
                x = self.layer3[2](x, residua=residual, subpartition=1)
                x = self.layer3[3:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 37:
                x = self.layer3[2](x, residua=residual, subpartition=2)
                x = self.layer3[3:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 38:
                x = self.layer3[2](x, residua=residual, subpartition=3)
                x = self.layer3[3:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block4
            elif partition == 39:
                x = self.layer3[3:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 40:
                x = self.layer3[3](x, residua=residual, subpartition=1)
                x = self.layer3[4:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 41:
                x = self.layer3[3](x, residua=residual, subpartition=2)
                x = self.layer3[4:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 42:
                x = self.layer3[3](x, residua=residual, subpartition=3)
                x = self.layer3[4:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block5
            elif partition == 43:
                x = self.layer3[4:](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 44:
                x = self.layer3[4](x, residua=residual, subpartition=1)
                x = self.layer3[5](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 45:
                x = self.layer3[4](x, residua=residual, subpartition=2)
                x = self.layer3[5](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 46:
                x = self.layer3[4](x, residua=residual, subpartition=3)
                x = self.layer3[5](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer3, block6
            elif partition == 47:
                x = self.layer3[5](x)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 48:
                x = self.layer3[5](x, residua=residual, subpartition=1)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 49:
                x = self.layer3[5](x, residua=residual, subpartition=2)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 50:
                x = self.layer3[5](x, residua=residual, subpartition=3)
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer4, block1
            elif partition == 51:
                x = self.layer4(x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 52:
                x = self.layer4[0](x, residua=residual, subpartition=1)
                x = self.layer4[1:](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 53:
                x = self.layer4[0](x, residua=residual, subpartition=2)
                x = self.layer4[1:](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 54:
                x = self.layer4[0](x, residua=residual, subpartition=3)
                x = self.layer4[1:](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer4, block2
            elif partition == 55:
                x = self.layer4[1:](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 56:
                x = self.layer4[1](x, residua=residual, subpartition=1)
                x = self.layer4[2](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 57:
                x = self.layer4[1](x, residua=residual, subpartition=2)
                x = self.layer4[2](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 58:
                x = self.layer4[1](x, residua=residual, subpartition=3)
                x = self.layer4[2](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            # partition layer4, block3
            elif partition == 59:
                x = self.layer4[2](x)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 60:
                x = self.layer4[2](x, residua=residual, subpartition=1)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 61:
                x = self.layer4[2](x, residua=residual, subpartition=2)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 62:
                x = self.layer4[2](x, residua=residual, subpartition=3)

                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 63:
                x = self.deconv_layers(x)
                x = self.final_layer(x)
            elif partition == 64:
                x = self.final_layer(x)
            elif partition == 65:
                x = x
        return x

resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get(cfg):
    num_layers = cfg.RESNET.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = ResNet(block_class, layers, cfg)
    return model