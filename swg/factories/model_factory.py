from gen import GeneratorCONV_MNIST, GeneratorDCGAN_CelebA
from disc import DiscriminatorCONV_MNIST, DiscriminatorDCGAN_CelebA


def get_model(identifier: str, device: str, use_disc: bool):
    resolvers = {
        'mnist': get_mnist_model,
        'celeba': get_celeba_model
    }
    return resolvers[identifier](device, use_disc)


def get_mnist_model(device: str, use_disc: bool):
    noise_dim = 32
    G = GeneratorCONV_MNIST(noise_dim).to(device)
    if use_disc:
        f_dim = 256
        D = DiscriminatorCONV_MNIST(784, f_dim).to(device)
        return G, D
    return G


def get_celeba_model(device: str, use_disc: bool):
    noise_dim = 100
    G = GeneratorDCGAN_CelebA(noise_dim).to(device)
    if use_disc:
        D = DiscriminatorDCGAN_CelebA(3).to(device)
        return G, D
    return G
