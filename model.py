import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class dCNN(nn.Module):
    def __init__(self, x_in: int, latent_dim: int, num_convs: int, activation):
        super(dCNN, self).__init__()
        self.conv_enc1 = nn.Conv3d(x_in,latent_dim,kernel_size=(3,3,3),padding='same',padding_mode='circular')
        self.activation = activation
        self.num_convs = num_convs
        self.layers = nn.ModuleList()
        for i in range(0,num_convs): 
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=1,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=2,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=4,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=8,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=4,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=2,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.Conv3d(latent_dim,latent_dim,kernel_size=(3,3,3),dilation=1,padding='same',padding_mode='circular'))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        self.conv_dec1 = nn.Conv3d(latent_dim,x_in,kernel_size=(3,3,3),padding='same',padding_mode='circular')
            
    def forward(self,x): #48
        x_inc = x
        x = self.activation(self.conv_enc1(x))
        for i in range(self.num_convs):
            x_0 = x
            for j in range(7):   
                x = self.activation(self.layers[7*i+j](x)) #0 to 27 i_max :3 j_max :6
            x = x + x_0
        x = self.conv_dec1(x)
        return x + x_inc
    


##### UNET 


class DoubleConv(nn.Module):
    """(convolution => [BN] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3,3,3), padding='same',padding_mode='circular'),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3,3,3), dilation=2, padding='same',padding_mode='circular'),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, mid_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,padding='same',padding_mode='circular')

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64,64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_channels))

    def forward(self, x):
        inp = x 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits 



