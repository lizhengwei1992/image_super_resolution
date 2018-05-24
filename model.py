import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import pdb

def Inter_Bicubic(x, scale):
    x_numpy = x.data.cpu().numpy()
    x_resize = np.random.random([x_numpy.shape[0],x_numpy.shape[1],x_numpy.shape[2]*scale,x_numpy.shape[3]*scale])

    for i in range(x_numpy.shape[0]):

        x_resize[i,0,:,:] = cv2.resize(x_numpy[i,0,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,1,:,:] = cv2.resize(x_numpy[i,1,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)
        x_resize[i,2,:,:] = cv2.resize(x_numpy[i,2,:,:], (x_numpy.shape[3]*scale,x_numpy.shape[2]*scale), interpolation=cv2.INTER_CUBIC)

    return  Variable(torch.from_numpy(x_resize).float().cuda(), volatile=False)

def Conv(nFeat_in, nFeat_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Conv2d(
        nFeat_in, nFeat_out, kernel_size=3,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def sub_patch(input, scale_factor):
    batch_size, in_channels, in_height, in_width = input.size()

    out_channels = int(in_channels // (scale_factor * scale_factor))
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(
            batch_size, out_channels, scale_factor, scale_factor,
            in_height, in_width)
        input_view = input_view.permute(0, 1, 3, 2, 4, 5).contiguous()
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(
            batch_size, in_channels, out_height, block_size,
            out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)
class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()

        modules = []
        
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)

        return x

class Conv_LReLU_Block(nn.Module):
    def __init__(self, nFeat_in, nFeat_out, kernel_size=3, act=nn.LeakyReLU(0.05)):
        super(Conv_LReLU_Block, self).__init__()

        modules = []
        modules.append(nn.Conv2d(
            nFeat_in, nFeat_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2))
        modules.append(act)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out

class Conv_LReLU(nn.Module):
    def __init__(self, nFeat_in, nFeat_out, kernel_size=3, act=nn.LeakyReLU(0.05)):
        super(Conv_LReLU, self).__init__()

        modules = []
        modules.append(nn.Conv2d(
            nFeat_in, nFeat_out, kernel_size=kernel_size, padding=(kernel_size-1) // 2))
        modules.append(act)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out



class ULSee_subimg(nn.Module):

    def forward(self, x):
        
        x = self.headConv(x)
        x = self.body(x)
        x = self.tailConv(x)
        us = self.upsample(x)

        output = us
    
    
        return output

class ULSee_bilinear(nn.Module):

    def forward(self, x):
        
        x_bilinear = self.bilinear(x)
        x = sub_patch(x, 0.5) 
        x = self.headConv(x)
        x = self.body(x)
        x = self.tailConv(x)
        us = self.upsample(x)

        output = us + x_bilinear
    
    
        return output

class ULSee_mcn(nn.Module):

    def forward(self, x):
        

        x = self.headConv(x)
        x = self.body(x)

        x = self.tailConv(x)
        us = sub_patch(x, 3)

        output =  us
    
        return output



class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):

        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y
        return out


class make_se_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_se_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.se = SELayer(nChannels)
    def forward(self, x):

        out = F.relu(self.conv(x))
        x = self.se(x)
        out = torch.cat((x, out), 1)
        return out

class RDseB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDseB, self).__init__()
        nChannels_ = nChannels
        modules = []
        nChannels_ = nChannels  + growthRate
        for i in range(nDenselayer):
            modules.append(make_se_dense(nChannels_, growthRate))
            nChannels_ = nChannels_  + growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nBlock = args.nBlock
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3 
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB4 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*6, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = self.RDB1(F_3)
        FF = torch.cat((F_1, F_2, F_3, F_4), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)
        output = self.conv3(us)

        return output

# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)


    def forward(self, x):

        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class RDN_v2(nn.Module):
    def __init__(self, args):
        super(RDN_v2, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer        
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel*4, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RFB 3 
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB4 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB5 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB6 = RDB(nFeat, nDenselayer, growthRate)
        # self.RDB7 = RDB(nFeat, nDenselayer, growthRate)
        # self.RDB8 = RDB(nFeat, nDenselayer, growthRate)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*6, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_down = nn.Conv2d(nFeat, args.nChannel*scale*scale*4, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale*2)

    def forward(self, x):

        x = sub_patch(x, 0.5)
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = self.RDB4(F_3)
        F_5 = self.RDB5(F_4)
        F_6 = self.RDB6(F_5)
        # F_7 = self.RDB7(F_6)
        # F_8 = self.RDB8(F_7)

        FGF = torch.cat((F_1, F_2, F_3, F_4, F_5, F_6), 1)
        FGF = self.GFF_1x1(FGF)
        FGF = self.GFF_3x3(FGF)
        FDF = FGF + F_
        output = self.conv_down(FDF)
        output = self.upsample(output)

        return output


class RDN_Tiny(nn.Module):
    def __init__(self, args):
        super(RDN_Tiny, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer        
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel*4, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RFB 3 
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB4 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB5 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB6 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB7 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB8 = RDB(nFeat, nDenselayer, growthRate)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*8, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_down = nn.Conv2d(nFeat, args.nChannel*scale*scale*4, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale*2)

    def forward(self, x):

        x = sub_patch(x, 0.5)
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = self.RDB4(F_3)
        F_5 = self.RDB5(F_4)
        F_6 = self.RDB6(F_5)
        F_7 = self.RDB7(F_6)
        F_8 = self.RDB8(F_7)

        FGF = torch.cat((F_1, F_2, F_3, F_4, F_5, F_6, F_7, F_8), 1)
        FGF = self.GFF_1x1(FGF)
        FGF = self.GFF_3x3(FGF)
        FDF = FGF + F_
        output = self.conv_down(FDF)
        output = self.upsample(output)

        return output