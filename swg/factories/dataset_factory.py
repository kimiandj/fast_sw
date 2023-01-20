from torchvision import transforms
from torchvision.datasets import MNIST, CelebA


def get_dataset(identifier: str, dataroot: str, train: bool):
    resolvers = {
        'mnist': get_mnist_dataset,
        'celeba': get_celeba_dataset
    }
    return resolvers[identifier](dataroot, train)


def get_mnist_dataset(dataroot: str, train: bool):
    dataset_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return MNIST(root=dataroot,
                 train=train,
                 download=True,
                 transform=dataset_transforms)


def get_celeba_dataset(dataroot: str, train: bool):
    dataset_transforms = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    split = 'train' if train else 'valid'
    return CelebA(root=dataroot,
                  split=split,
                  download=True,
                  transform=dataset_transforms)
