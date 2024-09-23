import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from Accuracy.src.utils.seeds import set_seed
set_seed(24)

def cifar_get10(batch_size, data_root='~/Documents/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def cifar_get100(batch_size, data_root='~/Documents/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size,train=True, val=True,number_workers = 1):
    # Data loading code
    imagenet_data = '/home/wangcong/dataset/ImageNet2012/imagenet'
    print("Building ImageNet data loader with {} workers".format(number_workers))
    traindir = os.path.join(imagenet_data, 'train')
    valdir = os.path.join(imagenet_data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=number_workers, pin_memory=True, sampler=train_sampler)
        ds.append(train_loader)
    if val:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=number_workers, pin_memory=True)
        ds.append(val_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_imagenet_partial(batch_size,train=True, val=True,number_workers = 1):
    # Data loading code
    imagenet_data = '/home/wangcong/dataset/ImageNet2012/imagenet50'
    # imagenet_data = '/hpc2hdd/home/cwang841/dataset/imagenet50/imagenet50'
    print("Building ImageNet data loader with {} workers".format(number_workers))
    traindir = os.path.join(imagenet_data, 'train')
    valdir = os.path.join(imagenet_data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=number_workers, pin_memory=True, sampler=train_sampler)
        ds.append(train_loader)
    if val:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=number_workers, pin_memory=True)
        ds.append(val_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def load_datasets(dataset_name,batch_size):
    assert dataset_name in ['Cifar10','Cifar100','ImageNetPartial','ImageNet'], dataset_name
    if dataset_name == 'Cifar10':
        train_loader, test_loader = cifar_get10(batch_size)
    elif dataset_name == 'Cifar100':
        train_loader, test_loader = cifar_get100(batch_size)
    elif dataset_name == 'ImageNetPartial':
        train_loader, test_loader = get_imagenet_partial(batch_size)
    elif dataset_name == 'ImageNet':
        train_loader, test_loader = get_imagenet(batch_size)
    else:
        train_loader, test_loader = None, None

    return train_loader,test_loader