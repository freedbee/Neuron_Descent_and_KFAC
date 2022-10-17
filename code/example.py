import os
import torch
import numpy as np
import argparse
import pickle
# Custom Imports
import architectures 
import dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("-n_ep", default=10,
                    type=int, help='number of epochs')
parser.add_argument("-subset", default=0,
                    type=int, help='determines whether training set is restricted to subset of 1000 images. Note that this choice overwrites the batch size.')
parser.add_argument("-dataset", default='fashion',
                    choices=['mnist', 'fashion', 'cifar10'],
                    type=str)
parser.add_argument("-batch_size", default=100,
                    type=int)
parser.add_argument("-optimizer", default='sgd',
                    type=str, 
                    choices=['sgd', 'adam', 'foof', 'kfac', 'natural', 'natural_bd'], 
                    help='bd is short for block diagonal')
parser.add_argument("-lr", default=1.,
                    type=float)
parser.add_argument("-damp", default=1.,
                    type=float)
parser.add_argument("-kf_mom", default=0.95,
                    type=float, help='momentum used for EMA of kronecker factors')
parser.add_argument("-sgd_mom", default=0.0,
                    type=float, help='standard momentum')
parser.add_argument("-inversion_period", default=1,
                    type=int)
parser.add_argument("-heuristic_damping", default=1,
                    type=int)
parser.add_argument("-kf_n_update_data", default=None)
parser.add_argument("-kf_n_update_steps", default=None)
parser.add_argument("-mc_fisher", default=1, type=int)
parser.add_argument("-seed", default=42,
                    type=int, help="What's your question?")
parser.add_argument("-arch", default='fc', type=str, 
                    choices=['fc', 'resnet'],
                    help='architecture, fully connected or ResNet18')
parser.add_argument("-resnet_bn", default=1, type=int,  
                    help='determines whether resnet uses batch norm')

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

measure_freq = 100 # how often are train stats printed

trainloader, testloader = dataloaders.get_dataloaders(args.dataset, 
                                                    batch_size=args.batch_size, 
                                                    subset=args.subset, 
                                                    num_workers=2)
num_classes = 10
    
width = 1000
depth = 3
input_dim = 784
output_dim = num_classes

if args.arch == 'fc':
    net = architectures.SimpleFCNet(width=width, 
                                    depth=depth, 
                                    input_dim=input_dim, 
                                    output_dim=output_dim, 
                                    )
elif args.arch == 'resnet':
    net = architectures.ResNet18(bn=args.resnet_bn)

if args.optimizer in ['foof', 'kfac', 'natural', 'natural_bd']:
    net.set_optimizer(args.optimizer)
    net.set_lr(args.lr)
    net.set_damp(args.damp)
    net.set_MC_fisher(args.mc_fisher)
    net.set_heuristic_damping(args.heuristic_damping)
    net.set_inversion_period(args.inversion_period)
    net.set_kf_m(args.kf_mom)
    if args.kf_n_update_data is None:
        net.set_kf_n_update_data(None)
    else:
        net.set_kf_n_update_data(int(args.kf_n_update_data))
    if args.kf_n_update_steps is None:
        net.set_kf_n_update_steps(None)
    else:
        net.set_kf_n_update_steps(int(args.kf_n_update_steps))
    net.set_output_dim(num_classes) # note that this isn't an issue for autoencoders
    net.set_momentum(0.0) 
    
net.to(device)
criterion = torch.nn.CrossEntropyLoss()


if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.sgd_mom)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    net.initialise_kf_fisher(trainloader)    



for epoch in range(args.n_ep):
    print('\n\nEpoch', epoch)
    for t, (X, y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        if t%measure_freq == 0:
            # disable hook before collecting train stats, and reset after
            fh = net.forward_hook
            net.set_hooks(forward_hook=False)
            loss = criterion(net(X), y)
            net.set_hooks(forward_hook=fh)
            do_print = 1
            if do_print:
                print(' pre update train loss', loss.item(), '   (mini-batch estimate)')         
        if args.optimizer in ['sgd', 'adam']:
            loss = criterion(net(X), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.optimizer in ['foof', 'kfac', 'natural', 'natural_bd']:
            net.parameter_update(X,y)
        
            if t%measure_freq == 0:
            # disable hook before collecting train stats, and reset after
                fh = net.forward_hook
                net.set_hooks(forward_hook=False)
                loss = criterion(net(X), y)
                net.set_hooks(forward_hook=fh)
                do_print = 1
                if do_print:
                    print('post update train loss', loss.item(), '   (mini-batch estimate)')    
            