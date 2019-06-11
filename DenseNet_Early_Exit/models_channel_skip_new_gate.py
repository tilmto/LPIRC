import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math

BatchNorm = nn.BatchNorm2d

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet100(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=24, growth_rate=12, block_config=(16, 16, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model




def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.num_input_features = num_input_features
        self.dense_module = nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(num_input_features)),
                      ('relu1', nn.ReLU(inplace=True)),
                      ('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
                      ('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
                      ('relu2', nn.ReLU(inplace=True)),
                      ('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))])
                       )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.dense_module(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

     
class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.trans_module = nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),
                       ('relu', nn.ReLU(inplace=True)),
                       ('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)),
                       ('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))
    def forward(self, x):
        return self.trans_module(x)






class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, block=BasicBlock):

        super(DenseNet, self).__init__()

        # First convolution
        self.growth_rate = growth_rate
        self.base_layer = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.block_config = block_config
        num_features = num_init_features
        
        

        # denseblock 0
        for i in range(block_config[0]):
            setattr(self, 'denseblock0_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))
            
            gate = NewGate(pool_size = 56, channel = num_features + (i + 1) * growth_rate, out_channels = growth_rate)
            
            setattr(self, 'denseblock0_%s_gate' % i, gate)
            
        num_features = (num_features + block_config[0] * growth_rate)
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 

        # denseblock 1
        for i in range(block_config[1]):
            setattr(self, 'denseblock1_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))
            
            gate = NewGate(pool_size = 28, channel = num_features + (i + 1) * growth_rate, out_channels = growth_rate)
            
            setattr(self, 'denseblock1_%s_gate' % i, gate)
            
        num_features = (num_features + block_config[1] * growth_rate)
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2      



        # denseblock 2
        for i in range(block_config[2]):
            setattr(self, 'denseblock2_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))
            
            gate = NewGate(pool_size = 14, channel = num_features + (i + 1) * growth_rate, out_channels = growth_rate)
            
            setattr(self, 'denseblock2_%s_gate' % i, gate)
            
        num_features = (num_features + block_config[2] * growth_rate)
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        # denseblock 3
        for i in range(block_config[3]):
            setattr(self, 'denseblock3_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))
            
            gate = NewGate(pool_size = 7, channel = num_features + (i + 1) * growth_rate, out_channels = growth_rate)
            
            setattr(self, 'denseblock3_%s_gate' % i, gate)
           
        num_features = (num_features + block_config[3] * growth_rate)


        # Final batch norm
        self.bn_norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        
        ## branch network1
        branch_channels = [32, 64, 32]
        
        self.branch_layer1 = nn.Sequential(
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(branch_channels[0],
                                branch_channels[1], kernel_size=5,
                                stride=1, padding=2,
                                bias=False),
                                BatchNorm(branch_channels[1]),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(branch_channels[1], branch_channels[2], kernel_size=3, stride=1, padding=1,
                                          bias=False),
                                BatchNorm(branch_channels[2]),
                                nn.AvgPool2d(14),
                                nn.Conv2d(branch_channels[2], num_classes, 
                              kernel_size=1,stride=1, padding=0, bias=True),
                                )
        
         ## branch network2
        branch_channels = [32, 64, 32]
        
        
        self.branch_layer2 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(7),
                              nn.Conv2d(branch_channels[2], num_classes, 
                              kernel_size=1,stride=1, padding=0, bias=True),
                              )
        
        ## branch network3
        branch_channels = [32, 64, 32]
        
        self.branch_layer3 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(7),
                              nn.Conv2d(branch_channels[2], num_classes, 
                              kernel_size=1,stride=1, padding=0, bias=True),
                              )
        
        ## branch network4
        branch_channels = [32, 64, 32]
        
        self.branch_layer4 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(7),
                              nn.Conv2d(branch_channels[2], num_classes, 
                              kernel_size=1,stride=1, padding=0, bias=True),
                              )
        
        ## branch network5
        branch_channels = [32, 64, 32]
        
        self.branch_layer5 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(7),
                              nn.Conv2d(branch_channels[2], num_classes, 
                              kernel_size=1,stride=1, padding=0, bias=True),
                              )
        
        ## branch network6
        branch_channels = [32, 64, 32]
        
        self.branch_layer6 = nn.Sequential(
                              nn.AvgPool2d(7),
                              View(-1, 32* block.expansion),
                              nn.Linear(32 * block.expansion, num_classes)
                              )
        
        
        ## branch network7
        self.branch_layer7 = nn.Sequential(
                              nn.AvgPool2d(7),
                              View(-1, 32* block.expansion),
                              nn.Linear(32 * block.expansion, num_classes)
                              )


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, front_layer_idx, back_layer_index, num_input_features, growth_rate, bn_size, drop_rate):
        modules = []
        for i in range(front_layer_idx, back_layer_index):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            modules.extend([layer])
        return nn.Sequential(*modules)


    def forward(self, x):
        features = self.base_layer(x)
        
        output_branch = []
        
        masks = []
        gprobs = []
        
        new_features = getattr(self, 'denseblock0_0')(features)
        
        features = torch.cat([features, new_features], 1)
        
        mask,gprob = getattr(self, 'denseblock0_0_gate')(features)
        
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        
        

        # loop for denseblock 0
        for i in range(1, self.block_config[0]):
            
            # print(mask.size())
            
            
            new_features = getattr(self, 'denseblock0_{}'.format(i))(features) 
            
            if i == int( self.block_config[0] * 1/2):
                x_branch1 = self.branch_layer1(new_features)
                x_branch1 = x_branch1.view(x_branch1.size(0), -1)
                output_branch.append(x_branch1)
            
            # print(new_features.size())
            
            
            new_features = mask.expand_as(new_features) * new_features
            
            
            features = torch.cat([features, new_features], 1)
            
            mask, gprob = getattr(self, 'denseblock0_{}_gate'.format(i))(features)
            
            gprobs.append(gprob)
            masks.append(mask.squeeze())
            
        features = self.trans0(features)
        


        # loop for denseblock 1
        for i in range(self.block_config[1]):
            
            new_features = getattr(self, 'denseblock1_{}'.format(i))(features) 
            new_features = mask.expand_as(new_features) * new_features
            
            if i == int( self.block_config[1] * 1/4):
                x_branch2 = self.branch_layer2(new_features)
                x_branch2 = x_branch2.view(x_branch2.size(0), -1)
                output_branch.append(x_branch2)
            if i == int( self.block_config[1] * 3/4):
                x_branch3 = self.branch_layer3(new_features)
                x_branch3 = x_branch3.view(x_branch3.size(0), -1)
                #print('The shape of x branch2 = ' + str(x_branch2.size()))
                output_branch.append(x_branch3)
            
            
            features = torch.cat([features, new_features], 1)
            
            mask, gprob = getattr(self, 'denseblock1_{}_gate'.format(i))(features)
            
            gprobs.append(gprob)
            masks.append(mask.squeeze())
            
        features = self.trans1(features)

        # loop for denseblock 2
        for i in range(self.block_config[2]):
            new_features = getattr(self, 'denseblock2_{}'.format(i))(features)  
            new_features = mask.expand_as(new_features) * new_features
            
            if i == int( self.block_config[2] * 1/4):
                x_branch4 = self.branch_layer4(new_features)
                x_branch4 = x_branch4.view(x_branch4.size(0), -1)
                output_branch.append(x_branch4)
            if i == int( self.block_config[2] * 3/4):
                x_branch5 = self.branch_layer5(new_features)
                x_branch5 = x_branch5.view(x_branch5.size(0), -1)
                #print('The shape of x branch2 = ' + str(x_branch2.size()))
                output_branch.append(x_branch5)
            
            
            features = torch.cat([features, new_features], 1)
            
            mask, gprob = getattr(self, 'denseblock2_{}_gate'.format(i))(features)
            
           
            gprobs.append(gprob)
            masks.append(mask.squeeze())
                
        features = self.trans2(features)

        # loop for denseblock 3
        for i in range(self.block_config[3]):
            new_features = getattr(self, 'denseblock3_{}'.format(i))(features) 
            new_features = mask.expand_as(new_features) * new_features
            
            if i == int( self.block_config[3] * 1/4):
                x_branch6 = self.branch_layer6(new_features)
                x_branch6 = x_branch6.view(x_branch6.size(0), -1)
                output_branch.append(x_branch6)
            if i == int( self.block_config[3] * 3/4):
                x_branch7 = self.branch_layer7(new_features)
                x_branch7 = x_branch7.view(x_branch7.size(0), -1)
                #print('The shape of x branch2 = ' + str(x_branch2.size()))
                output_branch.append(x_branch7)
            
            features = torch.cat([features, new_features], 1)
            
            mask, gprob = getattr(self, 'denseblock3_{}_gate'.format(i))(features)
            
            if i < self.block_config[3] - 1:
                
                gprobs.append(gprob)
                masks.append(mask.squeeze())
            

        features = self.bn_norm(features)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out,masks,gprobs,output_branch
    

########################################
# DenseNet with New Gate     #
########################################

class NewGate(nn.Module):
    """ one 1x1 conv followed by a 3x3 conv (stride=1) layer """
    def __init__(self, pool_size=5, channel=10, out_channels = 32):
        super(NewGate, self).__init__()
        self.pool_size = pool_size
        self.channel = channel
        self.activate = False
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        pool_size = math.floor(pool_size/2 + 0.5)
        
        self.avg_layer = nn.AvgPool2d(pool_size)

        self.linear_layer = nn.Conv2d(in_channels=out_channels, out_channels=32,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Sigmoid()
        self.logprob = nn.LogSoftmax()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        
        x = self.avg_layer(x)
        
        x = self.linear_layer(x)
        
        x = x.view(x.size(0), -1)

        prob = self.prob_layer(x)
        logprob = self.logprob(x)
        # discretize
        x = (prob > 0.5).float().detach() - \
            prob.detach() + prob
        
        x = x.view(x.size(0), -1, 1, 1)
        return x, logprob
    
    
    
########################################
# DenseNet with Feedforward Gate     #
########################################
# FFGate-II
class FeedforwardGateII(nn.Module):
    """ use single conv (stride=2) layer only"""
    def __init__(self, pool_size=5, channel=10, out_channels = 2):
        super(FeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel
        self.activate = False
        self.energy_cost = 0
        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2 + 0.5) # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=out_channels,
                                      kernel_size=1, stride=1)
        # self.prob_layer = nn.Softmax()
        self.prob_layer = nn.Sigmoid()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)

        x = self.linear_layer(x)
        
        # print(x.size())
        
        x = x.view(x.size(0), -1)

        prob = self.prob_layer(x)
        logprob = self.logprob(x)
        # discretize
        x = (prob > 0.5).float().detach() - \
            prob.detach() + prob
        
        x = x.view(x.size(0), -1, 1, 1)
        return x, logprob

