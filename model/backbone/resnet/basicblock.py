import torch.nn as nn
import ipdb


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, dilation=dilation)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, dilation=(1, dilation, dilation))


def conv1x3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    if isinstance(stride, int):
        stride_1, stride_2, stride_3 = 1, stride, stride
    else:
        stride_1, stride_2, stride_3 = 1, stride[1], stride[2]

    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                     stride=(stride_1, stride_2, stride_3),
                     padding=(0, 1, 1), bias=False, dilation=(1, dilation, dilation))


def conv1x3x3_conv3x1x1(in_planes, out_planes, stride=1, dilation=1, nb_temporal_conv=3):
    "3x3 convolution with padding"
    if isinstance(stride, int):
        stride_2d, stride_1t = (1, stride, stride), (stride, 1, 1)
    else:
        stride_2d, stride_1t = (1, stride[1], stride[2]), (stride[0], 1, 1)

    _2d = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride_2d,
                    padding=(0, 1, 1), bias=False, dilation=dilation)

    _1t = nn.Sequential()
    for i in range(nb_temporal_conv):
        temp_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=stride_1t,
                              padding=(1, 0, 0), bias=False, dilation=1)
        _1t.add_module('temp_conv_{}'.format(i), temp_conv)
        _1t.add_module(('relu_{}').format(i), nn.ReLU(inplace=True))

    return _2d, _1t


class BasicBlock(nn.Module):
    expansion = 1
    only_2D = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv1, self.conv2 = None, None
        self.input_dim = 5
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock3D(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.conv2 = conv3x3x3(planes, planes, dilation)


class BasicBlock2D(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        # not the same input size here to speed up training
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.conv2 = conv3x3(planes, planes, dilation)
        self.input_dim = 4


class BasicBlock2_1D(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, nb_temporal_conv=1):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv1, self.conv1_1t = conv1x3x3_conv3x1x1(inplanes, planes, stride, dilation,
                                                        nb_temporal_conv=nb_temporal_conv)
        self.conv2, self.conv2_1t = conv1x3x3_conv3x1x1(planes, planes, dilation,
                                                        nb_temporal_conv=nb_temporal_conv)


    def forward(self, x):
        residual = x

        out = self.conv1(x)  # 2D in space
        out = self.bn1(out)
        out = self.relu(out)

        # ipdb.set_trace()
        out = self.conv1_1t(out)  # 1D in time + relu after each conv

        out = self.conv2(out)  # 2D in space
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        out = self.conv2_1t(out)  # 1D in time

        return out
