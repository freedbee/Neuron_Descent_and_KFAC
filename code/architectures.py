import torch
import torch.nn as nn
import torch.nn.functional as F
import optimization_modules as om



def FCNet(layer_widths, nonlinearity='relu', bias=False, **kwargs):
    """"
    returns a fully connected network. 
    - "layer_widths" should be a list of integers 
        determining the widths of each layer.
        For an example, see "simple_fc_net()" below.
    """
    mods = []
    for i in range(len(layer_widths)-1):
        mods.append(om.FOOFLinear(layer_widths[i], layer_widths[i+1],\
            bias=bias, nonlinearity=nonlinearity))
        if nonlinearity=='relu' and (i != len(layer_widths)-2):
            mods.append(nn.ReLU())
        elif nonlinearity=='softplus' and (i != len(layer_widths)-2):
            mods.append(nn.Softplus())
        elif nonlinearity=='linear' or (i == len(layer_widths)-2):
            pass
        else:
            print('UNSUPPORTED NONLINEARITY. Using linear network.')
    return om.FOOFSequential(mods, **kwargs)

def SimpleFCNet(width=1000, depth=3, input_dim=784, **kwargs):
    """"
    returns a fully connected net with ReLU nonlinearity and without biases.
        - 'width' determines how many neurons each hidden layer has.
        - 'depth' determines how many hidden layers there are.
    """
    arch = [input_dim, *([width]*depth), kwargs['output_dim']]
    return FCNet(arch, nonlinearity='relu', bias=False, **kwargs)

############################
### CODE FOR RESNET BELOW
############################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True):
        super(BasicBlock, self).__init__()
        self.apply_bn = bn
        self.conv1 = om.FOOFConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        if self.apply_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = om.FOOFConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.apply_bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            short_list = [om.FOOFConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False)]
            if self.apply_bn:
                short_list += [nn.BatchNorm2d(self.expansion * planes)]
            self.shortcut = nn.Sequential(*short_list)
           
    def forward(self, x):
        if self.apply_bn:
            out = F.relu(self.bn1(self.conv1(x)),inplace=True)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out,inplace=True)
        else:
            out = F.relu((self.conv1(x)))#,inplace=True)
            out = (self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)#,inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn=True):
        super(Bottleneck, self).__init__()
        self.apply_bn = bn
        self.conv1 = om.FOOFConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = om.FOOFConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = om.FOOFConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            short_list = [om.FOOFConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False)]
            if self.apply_bn:
                short_list += [nn.BatchNorm2d(self.expansion * planes)]
            self.shortcut = nn.Sequential(*short_list)

    def forward(self, x):
        if self.apply_bn:
            out = F.relu(self.bn1(self.conv1(x)),inplace=True)
            out = F.relu(self.bn2(self.conv2(out)),inplace=True)
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out,inplace=True)
        else:
            out = F.relu((self.conv1(x)))#,inplace=True)
            out = F.relu((self.conv2(out)))#,inplace=True)
            out = (self.conv3(out))
            out = out + self.shortcut(x)
            out = F.relu(out)#,inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mini=False, bn=True):
        super(ResNet, self).__init__()
        self.apply_bn = bn
        self.in_planes = 64
        self.mod_list = []
        self.mod_list += [om.FOOFConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)]
        if self.apply_bn:
            self.mod_list += [nn.BatchNorm2d(64)]
        self.mod_list += self._make_layer(block, 64, num_blocks[0], stride=1, bn=bn)
        self.mod_list += self._make_layer(block, 128, num_blocks[1], stride=2, bn=bn)
        self.mod_list += self._make_layer(block, 256, num_blocks[2], stride=2, bn=bn)
        self.mod_list += self._make_layer(block, 512, num_blocks[3], stride=2, bn=bn)
        self.mod_list += [nn.AvgPool2d(4)]
        self.mod_list += [nn.Flatten()]
        self.mod_list += [om.FOOFLinear(512 * block.expansion, num_classes)]
    
    def _make_layer(self, block, planes, num_blocks, stride, bn):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn))
            self.in_planes = planes * block.expansion
        return layers


def ResNet18(bn=True, num_classes=10):
    a = ResNet(BasicBlock, [2, 2, 2, 2], bn=bn, num_classes=num_classes)
    return om.FOOFSequential(a.mod_list)
