import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 5:
        (n_out, n_in, length, height, width) = layer.weight.size()
        n = n_in * length * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_hidden(layer):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1, 1, layer.hidden_size), torch.zeros(layer.batch_size(), 1, layer.hidden_size))


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

######################
class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None, alpha=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
	self.alpha = alpha

    def perturb(self, X_nat, y, epsilons=None, cuda=False):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = move_data_to_gpu(torch.from_numpy(X), cuda)
        y_var = move_data_to_gpu(torch.from_numpy(y), cuda)
        X_var.requires_grad = True

        output, _ = self.model(X_var)
        loss = F.nll_loss(output, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, X_nat - self.alpha, X_nat + self.alpha) # not (0, 1)

        return X

######################



class BaselineCnn(nn.Module):
    def __init__(self, classes_num):
        super(BaselineCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        #self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
        #                       kernel_size=(5, 5), stride=(1, 1),
        #                       padding=(2, 2), bias=False)

        #self.conv5 = nn.Conv2d(in_channels=512, out_channels=256,
        #                       kernel_size=(5, 5), stride=(1, 1),
        #                       padding=(2, 2), bias=False)

        self.fc1 = nn.Linear(239616, classes_num, bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        #self.bn4 = nn.BatchNorm2d(512)
        #self.bn5 = nn.BatchNorm2d(256)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        #init_layer(self.conv4)
        #init_layer(self.conv5)
        init_layer(self.fc1)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        #init_bn(self.bn4)
        #init_bn(self.bn5)

    def forward(self, input, return_bottleneck=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        #x = F.relu(self.bn4(self.conv4(x)))
        #x = F.max_pool2d(x, kernel_size=(2, 2))
        #x = F.relu(self.bn5(self.conv5(x)))
        #x = F.max_pool2d(x, kernel_size=(2, 2))

        #x = F.max_pool2d(x, kernel_size=x.shape[2:])
        #x = x.view(x.shape[0:2])

	x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = F.log_softmax(self.fc1(x), dim=-1)

        return x




#####################################################################################################
class EmbeddingLayers(nn.Module):
    def __init__(self):
        super(EmbeddingLayers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        a1 = F.relu(self.bn1(self.conv1(x)))
        a1 = F.max_pool2d(a1, kernel_size=(2, 2))
        a2 = F.relu(self.bn2(self.conv2(a1)))
        a2 = F.max_pool2d(a2, kernel_size=(2, 2))
        a3 = F.relu(self.bn3(self.conv3(a2)))
        a3 = F.max_pool2d(a3, kernel_size=(2, 2))
        emb = F.relu(self.bn4(self.conv4(a3)))
        emb = F.max_pool2d(emb, kernel_size=(2, 2))

        if return_layers is False:
            return emb
        else:
            return [a1, a2, a3, emb]

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers()
        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input)

        # (samples_num, 512, hidden_units)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, x


#######################################################################################ResNet#################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):  # Res18: BasicBlock [2, 2, 2, 2]
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
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
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

    def forward(self, x):
        (_, seq_len, mel_bins) = x.shape

        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)
	
	output = F.log_softmax(output, dim=-1)

        return output, x

###########################################################################VGG16###########################################################
class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x

class VggishConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x


class Vggish(nn.Module):
    def __init__(self, classes_num):
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock3(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock3(in_channels=256, out_channels=512)
        self.conv_block5 = VggishConvBlock3(in_channels=512, out_channels=512)

        self.fc_final = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, x
