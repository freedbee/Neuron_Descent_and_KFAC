import torch
import torchvision
from torchvision import datasets, transforms

def get_dataloaders(dataset, batch_size=100, subset=False, num_workers=0):
    """"
    returns three dataloaders. the first two load the training set, 
    the third loads the testset.
        - 'dataset' should be one of mnist, fashion, cifar10. 
           Note that mnist/fashion are automatically flattened, while cifar10 isn't.
        - 'batch_size' determines batch size for all three dataloaders
        - 'subset' determines whether a subset of 1000 trainining images is used.
            If 'subset'is true, the batch_size is adjusted to 1000 automatically.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(784)
    ])
    transform_train = transform
    transform_test = transform
    if dataset == 'mnist':
        trainset = datasets.MNIST('../pytorch_data', train=True, download=True,
                            transform=transform_train)
        testset = datasets.MNIST('../pytorch_data', train=False,
                            transform=transform_test)
    if dataset == 'fashion':
        trainset = datasets.FashionMNIST('../pytorch_data', train=True, download=True,
                            transform=transform_train)

        testset = datasets.FashionMNIST('../pytorch_data', train=False,
                            transform=transform_test)
    
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        trainset = torchvision.datasets.CIFAR10(root='../pytorch_data', train=True, download=True,
                                        transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../pytorch_data', train=False, download=True,
                                        transform=transform_test)

    if subset:
        h = 1000
        trainset, _ = torch.utils.data.random_split(trainset, [h,59000], generator=torch.Generator().manual_seed(42))
        batch_size = h
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
            pin_memory=False, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
            pin_memory=False, drop_last=False)

    return trainloader, testloader