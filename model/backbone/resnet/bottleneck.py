import torch.nn as nn
import ipdb


class Bottleneck(nn.Module):
    expansion = 4
    only_2D = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = None
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.input_dim = 5
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dilation=(1, dilation, dilation))


class Bottleneck2D(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        # to speed up the inference process
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.input_dim = 4

        if isinstance(stride, int):
            stride_1, stride_2 = stride, stride
        else:
            stride_1, stride_2 = stride[0], stride[1]

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride_1, stride_2),
                               padding=(1, 1), bias=False)


class Bottleneck2_1D(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, nb_temporal_conv=1):
        super().__init__(inplanes, planes, stride, downsample, dilation)

        if isinstance(stride, int):
            stride_2d, stride_1t = (1, stride, stride), (stride, 1, 1)
        else:
            stride_2d, stride_1t = (1, stride[1], stride[2]), (stride[0], 1, 1)

        # CONV2
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride_2d,
                               padding=(0, dilation, dilation), bias=False, dilation=dilation)

        self.conv2_1t = nn.Sequential()
        for i in range(nb_temporal_conv):
            temp_conv = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=stride_1t,
                                  padding=(1, 0, 0), bias=False, dilation=1)
            self.conv2_1t.add_module('temp_conv_{}'.format(i), temp_conv)
            self.conv2_1t.add_module(('relu_{}').format(i), nn.ReLU(inplace=True))


    def forward(self, x):
        residual = x

        ## CONV1 - 3D (1,1,1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        ## CONV2
        #  Spatial - 2D (1,3,3)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Temporal - 3D (3,1,1)
        out = self.conv2_1t(out)

        ## CONV3 - 3D (1,1,1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
