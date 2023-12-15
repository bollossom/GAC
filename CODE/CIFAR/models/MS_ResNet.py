import torch.nn as nn
import torch
import torch
from models.GAU import *
from models.layers import *

class GAC(nn.Module):
    def __init__(self, T, out_channels):
        super().__init__()
        self.TA = TA(T=T)
        self.SCA = SCA(in_planes=out_channels, kerenel_size=4)  # 34 K=8#18K=4
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        x_seq = x_seq.permute(1, 0, 2, 3, 4)
        spikes = spikes.permute(1, 0, 2, 3, 4)
        # x_seq B T C H W
        # spikes B T inplanes H W
        # x_seq_2 B T inplanes H W
        x_seq_2 = x_seq
        TA = self.TA(x_seq_2)
        SCA = self.SCA(x_seq_2)
        out = self.sigmoid(TA *SCA)
        y_seq = out * spikes
        y_seq = y_seq.permute(1, 0, 2, 3, 4)
        return y_seq
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_MS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_MS, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = SeqToANNContainer(norm_layer(planes))
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = SeqToANNContainer(norm_layer(planes))
        self.downsample = downsample
        self.stride = stride
        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()

    def forward(self, x):
            identity = x
            out = self.spike1(x)
            out = self.bn1(self.conv1(out))
            out = self.spike2(out)
            out = self.bn2(self.conv2(out))

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

            return out


class ResNet(nn.Module):
    def __init__(self, block, layers, T=6,num_classes=10,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = SeqToANNContainer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False))

        self.bn1 = SeqToANNContainer(norm_layer(self.inplanes))

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = SeqToANNContainer((nn.AdaptiveAvgPool2d((1, 1))))

        self.fc1 = SeqToANNContainer(nn.Linear(512*block.expansion, num_classes))

        self.spike = LIFSpike()
        self.T = T
        self.encoding = GAC(T=self.T, out_channels=64)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SeqToANNContainer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x B C H W
        '''encoding'''
        x = self.conv1(x)
        x = self.bn1(x)
        img = x
        spike = self.spike(x)
        output = self.encoding(img, spike)
        '''encoding'''
        x = self.layer1(output)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spike(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = x.repeat(self.T,1,1,1,1)
        return self._forward_impl(x).permute(1,0,2)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def msresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock_MS, [3, 3, 2],
                   **kwargs)


if __name__ == '__main__':

    model = msresnet18(num_classes=10, T=6)
    x = torch.rand(2, 3, 32, 32)
    # print(model)
    y = model(x)
    print(y.shape)
    print("Parameter numbers: {}".format(
        sum(p.numel() for p in model.parameters())))







