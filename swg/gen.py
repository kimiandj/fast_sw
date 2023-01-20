import torch.nn as nn


class GeneratorCONV_MNIST(nn.Module):
    def __init__(self, nz):
        super(GeneratorCONV_MNIST, self).__init__()
        self.l1 = nn.Linear(nz, 1024)
        
        self.network = nn.Sequential(
          nn.ConvTranspose2d(1024, 64, 3, 2, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(64, 64, 3, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(64, 32, 3, 2, bias=False),
          nn.BatchNorm2d(32),
          nn.ReLU(True),
            
          nn.ConvTranspose2d(32, 32, 3, 1, bias=False),
          nn.BatchNorm2d(32),
          nn.ReLU(True),
            
          nn.ConvTranspose2d(32, 16, 2, 2, bias=False),
          nn.BatchNorm2d(16),
          nn.ReLU(True),
            
          nn.ConvTranspose2d(16, 1, 3, 1, bias=False),
          nn.Sigmoid()
      )
  
    def forward(self, input):
        output = self.l1(input)
        output = self.network(output.view(-1, 1024, 1, 1))
        return output


class GeneratorMNIST(nn.Module):
    def __init__(self, d):
        super(GeneratorMNIST, self).__init__()
        self.d = d
        
        self.l1 = nn.Sequential(nn.Linear(d, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l2 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l3 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l4 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l5 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l6 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
            
        self.l7 = nn.Sequential(nn.Linear(512, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU())
        
        self.l8 = nn.Sequential(nn.Linear(512, 784),
                                nn.Sigmoid())
        
    def forward(self, z):
        x = self.l1(z)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        return x


class GeneratorDCGAN_CelebA(nn.Module):
    def __init__(self, d):
        super(GeneratorDCGAN_CelebA, self).__init__()

        self.l1 = nn.Sequential(nn.Linear(d, 4 * 4 * 1024),
                                nn.BatchNorm1d(4 * 4 * 1024),
                                nn.ReLU())

        self.l2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU())

        self.l3 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU())

        self.l4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

        self.l5 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

        self.l6 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 2, bias=False),
                                nn.Sigmoid())

    def forward(self, z):
        x = self.l1(z)
        x = self.l2(x.view(-1, 1024, 4, 4))
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x
