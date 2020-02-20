# define AssistSR
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorConcatSkip2SR(nn.Module):
    def __init__(self):
        super(GeneratorConcatSkip2SR, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = 32
        nfc = N
        nc_im = 3
        ker_size = 3
        padd_size = 1
        # num_layer = 5
        num_layer = 3
        min_nfc = 32
        self.head = ConvBlock(nc_im, N, ker_size, padd_size,
                              1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.body = nn.Sequential()
        for i in range(1):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            # nn.Conv2d(max(N,min_nfc),nc_im,kernel_size=ker_size,stride =1,padding=padd_size),
            nn.ConvTranspose2d(max(N, min_nfc), 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            # nn.ConvTranspose2d(max(N,min_nfc), nc_im, kernel_size=5, stride=2, padding=1, output_padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        # x = self.up4(x)
        x = self.body(x)
        x = self.tail(x)
        # x = self.up2(x)
        # ind = int((y.shape[2]-x.shape[2])/2)
        # y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x


def netGSR():
    netSingG = GeneratorConcatSkip2SR()
    netSingG.apply(weights_init)
    return netSingG


netSR_my = netGSR().type(dtype)
netSR_my.load_state_dict(torch.load('/content/drive/My Drive/SR100_my.pth'))
netSR_my1 = netGSR().type(dtype)
netSR_my1.load_state_dict(torch.load('/content/drive/My Drive/SR100_faceAndPIE_gradient.pth'))
inputX = Variable(torch.ones(1, 3, 128, 128)).type(dtype)
print(netSR_my(inputX).shape)