import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


class RAMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform

        self.data = []
        self.targets = []
        for i in range(len(dataset)):
            self.data.append(dataset[i][0])
            self.targets.append(dataset[i][1])

    def __getitem__(self, index):

        if self.transform is not None:
            img = self.transform(self.data[index])
        else:
            img = self.data[index]
        return img, self.targets[index]
        
    def __len__(self):
        return len(self.data)

# prepare training loaders and test loaders for MNIST
def get_fmnist_loaders(batch_size, ram=True):
    trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                        (0.2860,), (0.3530,))
                                                 ]))

    testset = torchvision.datasets.FashionMNIST('./data', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.2860,), (0.3530,))
                                                ]))

    validset = testset

    transform = torchvision.transforms.RandomCrop(28, padding=2)

    if ram:
        trainset, validset, testset = RAMDataset(
            trainset), RAMDataset(validset), RAMDataset(testset)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader



def get_mnist_loaders(batch_size, ram=True):
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))

    testset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

    validset = testset

    transform = torchvision.transforms.RandomCrop(28, padding=2)

    if ram:
        trainset, validset, testset = RAMDataset(
            trainset), RAMDataset(validset), RAMDataset(testset)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def get_kmnist_loaders(batch_size, ram=True):
    trainset = torchvision.datasets.KMNIST('./data', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                (0.1918,), (0.3483,))
                                          ]))

    testset = torchvision.datasets.KMNIST('./data', train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                (0.1918,), (0.3483,))
                                         ]))

    validset = testset

    transform = torchvision.transforms.RandomCrop(28, padding=2)

    if ram:
        trainset, validset, testset = RAMDataset(
            trainset), RAMDataset(validset), RAMDataset(testset)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def get_cifar_loaders(batch_size, cifar=10, resize=32, ram=False, val=True):
    if cifar == 10:
        trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                        torchvision.transforms.Resize(resize),
                                                    torchvision.transforms.RandomCrop(32, padding=4),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                 ]))
        trainset, validset = torch.utils.data.random_split(
            trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
        testset = torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                        torchvision.transforms.Resize(resize)
                                                ]))
    elif cifar == 100:
        trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                                        torchvision.transforms.Resize(resize),
                                                        torchvision.transforms.RandomCrop(32, padding=4),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                  ]))
        trainset, validset = torch.utils.data.random_split(
            trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
        testset = torchvision.datasets.CIFAR100('./data', train=False, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                                        torchvision.transforms.Resize(resize)
                                                 ]))
    else:
        raise ValueError("cifar must be 10 or 100")

    if ram:

        if cifar == 10:
            trainset = RAMDataset(torchvision.datasets.CIFAR10('./data', train=True, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                        torchvision.transforms.Resize(resize),
                                                        torchvision.transforms.RandomCrop(32, padding=4),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                    ]))
            validset = RAMDataset(torchvision.datasets.CIFAR10('./data', train=True, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                        torchvision.transforms.Resize(resize),
                                                    ]))
            testset = RAMDataset(torchvision.datasets.CIFAR10('./data', train=False, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                            torchvision.transforms.Resize(resize)
                                                    ]))
            

        elif cifar == 100:
            trainset = RAMDataset(torchvision.datasets.CIFAR100('./data', train=True, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                                        torchvision.transforms.Resize(resize),
                                                        torchvision.transforms.RandomCrop(32, padding=4),
                                                        torchvision.transforms.RandomHorizontalFlip(),
                                                    ]))
            validset = RAMDataset(torchvision.datasets.CIFAR100('./data', train=True, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                                        torchvision.transforms.Resize(resize),
                                                    ]))
            testset = RAMDataset(torchvision.datasets.CIFAR100('./data', train=False, download=True),
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                                                            torchvision.transforms.Resize(resize)
                                                    ]))

    if val:
        trainset, _ = torch.utils.data.random_split(
            trainset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
        _, validset = torch.utils.data.random_split(
            validset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    else:
        validset = testset


    train_loader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                                    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, valid_loader, test_loader