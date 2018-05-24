import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data


def YUV_np2Tensor(imgIn, imgTar):
    ts = (2, 0, 1)
    w, h = imgIn.shape
    tensor_in = torch.Tensor(1,w,h)
    w,h = imgTar.shape
    tensor_tar = torch.Tensor(1,w,h)
    tensor_in[0,:,:] = torch.Tensor(imgIn.astype(float)).mul_(1.0)
    tensor_tar[0,:,:] = torch.Tensor(imgTar.astype(float)).mul_(1.0)
    

    return tensor_in, tensor_tar
def RGB_np2Tensor(imgIn, imgTar):
    ts = (2, 0, 1)

    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
    
    return imgIn, imgTar
def YCrCb_np2Tensor(imgIn, imgTar):
    ts = (2, 0, 1)

    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
    
    return imgIn, imgTar


def sharpen(im):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im_sharpen = cv2.filter2D(im, -1, kernel) / 2 + im / 2
    im_sharpen[im_sharpen > 255] = 255
    im_sharpen[im_sharpen < 0] = 0
    im_sharpen = im_sharpen.astype(np.uint8)
    return im_sharpen

def augment(imgIn, imgTar):
    if random.random() < 0.5:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
    if random.random() < 0.5:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]
    return imgIn, imgTar


def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)


    tp = args.patchSize
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]

    return imgIn, imgTar


class WX(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.scale = args.scale

        apath = args.dataDir

        dirHR = 'no_compress'
        dirLR = 'compress'
        self.dirIn = os.path.join(apath, dirLR)
        self.dirTar = os.path.join(apath, dirHR)

        self.fileList= os.listdir(self.dirIn)

        self.nTrain = len(self.fileList)

        print('file number %d' % len(self.fileList))

    def __getitem__(self, idx):

        scale = self.scale

        nameIn, nameTar = self.getFileName(idx)
        imgIn = cv2.imread(nameIn)
        imgTar = cv2.imread(nameTar)

        if self.args.need_patch:
            imgIn, imgTar = getPatch(imgIn, imgTar, self.args, scale)
        if self.args.blur:
            imgIn = cv2.GaussianBlur(imgIn, (5,5), 0.5)

        imgIn, imgTar = augment(imgIn, imgTar) 

        if self.args.YCrCb:
            imgIn = cv2.cvtColor(imgIn, cv2.COLOR_BGR2YCrCb)
            imgTar = cv2.cvtColor(imgTar, cv2.COLOR_BGR2YCrCb)

            return YCrCb_np2Tensor(imgIn, imgTar)
        else:
            if self.args.Y:
                imgIn = cv2.cvtColor(imgIn, cv2.COLOR_BGR2YCrCb)[:,:,0]
                imgTar = cv2.cvtColor(imgTar, cv2.COLOR_BGR2YCrCb)[:,:,0]                
                return YUV_np2Tensor(imgIn, imgTar)
            else:
                return RGB_np2Tensor(imgIn, imgTar)

    def __len__(self):

        return self.nTrain
      

    def getFileName(self, idx):

        name = self.fileList[idx]
        nameIn = os.path.join(self.dirIn, name)
        name = name[0:-4] + '.png'
        nameTar = os.path.join(self.dirTar, name)

        return nameIn, nameTar
