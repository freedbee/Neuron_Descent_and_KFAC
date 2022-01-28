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


