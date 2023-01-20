import torch.nn as nn


class DiscriminatorCONV_MNIST(nn.Module):
    def __init__(self, x_dim, f_dim):
        super(DiscriminatorCONV_MNIST, self).__init__()

        self.l1 = nn.Sequential(
                    nn.Linear(x_dim, f_dim),
                    nn.ReLU(True))
        
        self.l2 = nn.Sequential(
                    nn.Linear(f_dim, f_dim),
                    nn.ReLU(True))
        
        self.l3 = nn.Sequential(
                    nn.Linear(f_dim, 1))
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.l1(input)
        output = self.l2(output)
        features = output.view(input.size(0), -1)
        score = self.l3(output)
        return score, features


class DiscriminatorDCGAN_CelebA(nn.Module):
    def __init__(self, C):
        super(DiscriminatorDCGAN_CelebA, self).__init__()
        
        self.C = C

        self.l1 = nn.Sequential(nn.Conv2d(self.C, 64, 4, 2, 1),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU())
        
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU())
        
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU())
        
        self.l4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1),
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU())
        
        self.l5 = nn.Sequential(nn.Linear(4 * 4 * 512, 1))

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        features = output.view(x.size(0), -1)
        score = self.l5(features)
        return score, features
