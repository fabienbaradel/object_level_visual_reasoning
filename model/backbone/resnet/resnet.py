import torch.nn as nn
import math
from model.backbone.resnet.basicblock import BasicBlock2D
from model.backbone.resnet.bottleneck import Bottleneck2D
from utils.other import transform_input
import ipdb
import torch
from utils.meter import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

K_1st_CONV = 3


class ResNet(nn.Module):
    def __init__(self, blocks, layers, num_classes=1000, str_first_conv='2D',
                 num_final_fm=4,
                 two_heads=False,
                 size_fm_2nd_head=7,
                 blocks_2nd_head=None,
                 pooling='avg',
                 nb_temporal_conv=1,
                 list_stride=[1, 2, 2, 2],
                 **kwargs):
        self.nb_temporal_conv = nb_temporal_conv
        self.size_fm_2nd_head = size_fm_2nd_head
        self.two_heads = two_heads
        self.inplanes = 64
        self.input_dim = 5  # from 5D to 4D tensor if 2D conv
        super(ResNet, self).__init__()
        self.num_final_fm = num_final_fm
        self.time = None
        self._first_conv(str_first_conv)
        self.relu = nn.ReLU(inplace=True)
        self.list_channels = [64, 128, 256, 512]
        self.list_inplanes = []
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer1
        self.layer1 = self._make_layer(blocks[0], self.list_channels[0], layers[0], stride=list_stride[0])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer1
        self.layer2 = self._make_layer(blocks[1], self.list_channels[1], layers[1], stride=list_stride[1])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer2
        self.layer3 = self._make_layer(blocks[2], self.list_channels[2], layers[2], stride=list_stride[2])
        self.list_inplanes.append(self.inplanes)  # store the inplanes after layer3
        self.layer4 = self._make_layer(blocks[3], self.list_channels[3], layers[3], stride=list_stride[3])
        self.avgpool, self.avgpool_space, self.avgpool_time = None, None, None
        self.fc_classifier = nn.Linear(512 * blocks[3].expansion, num_classes)
        self.out_dim = 5
        self.pooling = pooling

        if self.two_heads:
            # Stride second head
            list_strides_2nd_head = self._get_stride_2nd_head()

            # Common block
            self.nb_block_common_trunk = 4 - len(blocks_2nd_head)

            self.list_layers_bis = []
            for i in range(self.nb_block_common_trunk, 4):
                # Take the correct inplanes
                self.inplanes = self.list_inplanes[i]

                # Create the layer
                layer = self._make_layer(blocks_2nd_head[i - self.nb_block_common_trunk],
                                         self.list_channels[i],
                                         layers[i],
                                         list_strides_2nd_head[i])
                if i == 0:
                    self.layer1_bis = layer
                    self.list_layers_bis.append(self.layer1_bis)
                elif i == 1:
                    self.layer2_bis = layer
                    self.list_layers_bis.append(self.layer2_bis)
                elif i == 2:
                    self.layer3_bis = layer
                    self.list_layers_bis.append(self.layer3_bis)
                elif i == 3:
                    self.layer4_bis = layer
                    self.list_layers_bis.append(self.layer4_bis)
                else:
                    raise NameError

            # List layers
            self.list_layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        # Pooling method
        if self.pooling == 'rnn':
            cnn_features_size = 512 * blocks[3].expansion
            hidden_state_size = 256 if cnn_features_size == 512 else 512
            self.rnn = nn.GRU(input_size=cnn_features_size,
                              hidden_size=hidden_state_size,
                              num_layers=1,
                              batch_first=True)
            self.fc_classifier = nn.Linear(hidden_state_size, num_classes)

        # Init of the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_stride_2nd_head(self):
        if self.size_fm_2nd_head == 7:
            return [1, 2, 2, 2]
        elif self.size_fm_2nd_head == 14:
            return [1, 2, 2, 1]
        elif self.size_fm_2nd_head == 28:
            return [1, 2, 1, 1]

    def _first_conv(self, str):
        self.conv1_1t = None
        self.bn1_1t = None
        if str == '3D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(K_1st_CONV, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                                   bias=False)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)


        elif str == '2.5D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                   bias=False)
            self.conv1_1t = nn.Conv3d(64, 64, kernel_size=(K_1st_CONV, 1, 1), stride=(1, 1, 1),
                                      padding=(1, 0, 0),
                                      bias=False)
            self.bn1_1t = nn.BatchNorm3d(64)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)

        elif str == '2D':
            self.conv1 = nn.Conv2d(3, 64,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(3, 3),
                                   bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.bn1 = nn.BatchNorm2d(64)
            self.input_dim = 4

        else:
            raise NameError

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        # Upgrade the stride is spatio-temporal kernel
        if not (block == BasicBlock2D or block == Bottleneck2D):
            stride = (1, stride, stride)

        if stride != 1 or self.inplanes != planes * block.expansion:
            if block is BasicBlock2D or block is Bottleneck2D:
                conv, batchnorm = nn.Conv2d, nn.BatchNorm2d
            else:
                conv, batchnorm = nn.Conv3d, nn.BatchNorm3d

            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False, dilation=dilation),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dilation, nb_temporal_conv=self.nb_temporal_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nb_temporal_conv=self.nb_temporal_conv))

        return nn.Sequential(*layers)

    def get_features_map(self, x, time=None, num=4, out_dim=None):
        if out_dim is None:
            out_dim = self.out_dim

        if self.time is None:
            B, C, T, W, H = x.size()
            self.time = T

        time = self.time

        # 5D -> 4D if 2D conv at the beginning
        x = transform_input(x, self.input_dim, T=time)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.conv1_1t is not None:
            x = self.conv1_1t(x)
            x = self.bn1_1t(x)
            x = self.relu(x)

        x = self.maxpool(x)

        # 1st residual block
        if num >= 1:
            # ipdb.set_trace()
            x = transform_input(x, self.layer1[0].input_dim, T=time)
            x = self.layer1(x)

        # 2nd residual block
        if num >= 2:
            x = transform_input(x, self.layer2[0].input_dim, T=time)
            x = self.layer2(x)

        # 3rd residual block
        if num >= 3:
            x = transform_input(x, self.layer3[0].input_dim, T=time)
            x = self.layer3(x)

        # 4th residual block
        if num >= 4:
            x = transform_input(x, self.layer4[0].input_dim, T=time)
            x = self.layer4(x)

        return transform_input(x, out_dim, T=time)

    def get_two_heads_feature_maps(self, x, T=None, out_dim=None, heads_type='object+context'):
        x = x['clip']  # (B, C, T, W, H)

        # Get the before last feature map
        x = self.get_features_map(x, T, num=self.nb_block_common_trunk)

        # Object head
        if 'object' in heads_type:
            fm_objects = x
            for i in range(len(self.list_layers_bis)):
                layer = self.list_layers_bis[i]
                fm_objects = transform_input(fm_objects, layer[0].input_dim, T=T)
                fm_objects = layer(fm_objects)
            fm_objects = transform_input(fm_objects, out_dim, T=T)
        else:
            fm_objects = None

        # Activity head
        if 'context' in heads_type:
            fm_context = x
            for i in range(self.nb_block_common_trunk, 4):
                layer = self.list_layers[i]
                fm_context = transform_input(fm_context, layer[0].input_dim, T=T)
                fm_context = layer(fm_context)
            fm_context = transform_input(fm_context, out_dim, T=T)
        else:
            fm_context = None

        return fm_context, fm_objects

    def forward(self, x):
        x = x['clip']

        x = self.get_features_map(x, num=self.num_final_fm)

        # Global average pooling
        if self.pooling == 'avg':
            self.avgpool = nn.AvgPool3d((x.size(2), x.size(-1), x.size(-1))) if self.avgpool is None else self.avgpool
            x = self.avgpool(x)
        elif self.pooling == 'rnn':
            self.avgpool_space = nn.AvgPool3d(
                (1, x.size(-1), x.size(-1))) if self.avgpool_space is None else self.avgpool_space
            x = self.avgpool_space(x)
            x = x.view(x.size(0), x.size(1), x.size(2))  # (B,D,T)
            x = x.transpose(1, 2)  # (B,T,D)
            ipdb.set_trace()
            x, _ = self.rnn(x)  # (B,T,D/2)
            x = torch.mean(x, 1)  # (B,D/2)

        # Final classif
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)

        return x
