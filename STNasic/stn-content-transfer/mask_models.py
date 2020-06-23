import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

debug = False
class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        if(debug):
            print("E1 pass")
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.BatchNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        if (debug):
            print("E2 pass")
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class D_B(nn.Module):
    def __init__(self, size):
        super(D_B, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
        )

    def forward(self, net, my_input):
        if (debug):
            print("D_B pass")
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)
        oimg = oimg * mask + my_input * (1 - mask)
        return oimg, mask

class D_B_removal(nn.Module):
    def __init__(self, size):
        super(D_B_removal, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
        )

    def forward(self, net, my_input, other_input, threshold):
        if (debug):
            print("start D_B pass")
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        mask = torch.sigmoid(output[:, :1])
        mask = mask.ge(threshold)
        mask = mask.type(torch.cuda.FloatTensor)
        mask = mask.repeat(1, 3, 1, 1)
        oimg = other_input * mask + my_input * (1 - mask)
        return oimg, mask

class D_A(nn.Module):
    def __init__(self, size):
        super(D_A, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        if (debug):
            print("D_A pass")
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net

class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        if (debug):
            print("Disc pass")
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net


class STN(nn.Module):
    def __init__(self, sep, size):
        super(STN, self).__init__()
        # self.conv0 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        # self.conv1 = nn.ConvTranspose2d(512, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.sep = sep
        self.size = size

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,  0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def toStn(self, x):
        if (debug):
            print("self.sep {}".format(self.sep))
            print("self.size {}".format(self.size))
            print("x shape {}".format(x.shape))
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))

        xs = xs.view(-1, 10 * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        #theta = torch.cat(32*[theta])
        if (debug):
            print("theta shape {}".format(theta.shape))
            print("x size {}".format(x.size()))

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        #x = x.view(-1, (512 - self.sep) * self.size * self.size)
        return x

    def stnTheta(self, x):
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))
        # xs = xs.view(-1, 1536 * 3 * 3)
        xs = xs.view(-1, 10 * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, x):
        if (debug):
            print("start stn pass")
        # transform the input
        x = self.toStn(x)
        return x

