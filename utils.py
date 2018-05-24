
import os
import os.path
import numpy as np

from math import exp
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class saveData():
	def __init__(self, args):
		self.args = args

		self.save_dir = os.path.join(args.saveDir, args.load)
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		self.save_dir_model = os.path.join(self.save_dir, 'model')
		if not os.path.exists(self.save_dir_model):
			os.makedirs(self.save_dir_model)

		if os.path.exists(self.save_dir + '/log.txt'):
			self.logFile = open(self.save_dir + '/log.txt', 'a')
		else:
			self.logFile = open(self.save_dir + '/log.txt', 'w')
			
	def save_model(self, model):
	    torch.save(
	        model.state_dict(),
	        self.save_dir_model + '/model_lastest.pt')
	    torch.save(
	        model,
	        self.save_dir_model + '/model_obj.pt')

	def save_log(self, log):
		self.logFile.write(log + '\n')

	def load_model(self, model):
		model.load_state_dict(torch.load(self.save_dir_model + '/model_lastest.pt'))
		print("load mode_status frmo {}/model_lastest.pt".format(self.save_dir_model))
		return model

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average = True, max_val = 255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)



# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
            
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
            
#             self.window = window
#             self.channel = channel


#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# class MS_SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(MS_SSIM, self).__init__()
#         self.size_average = size_average
#         self.channel = 3

#     def _ssim(self, img1, img2, size_average = True):
#         _, c, w, h = img1.size()
#         window_size = min(w, h, 11)
#         sigma = 1.5 * window_size / 11
#         window = create_window(window_size, sigma, self.channel).cuda()
#         mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
#         mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1*mu2

#         sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
#         sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

#         C1 = (0.01*255)**2
#         C2 = (0.03*255)**2
#         V1 = 2.0 * sigma12 + C2
#         V2 = sigma1_sq + sigma2_sq + C2
#         ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
#         mcs_map = V1 / V2
#         if size_average:
#             return ssim_map.mean(), mcs_map.mean()

#         def ms_ssim(self, img1, img2, levels=5):
            
#             weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

#             msssim = Variable(torch.Tensor(5,).cuda())
#             mcs = Variable(torch.Tensor(5,).cuda())
#             for i in range(5):
#                 ssim_map, mcs_map = self._ssim(img1, img2)
#                 msssim[i] = ssim_map
#                 mcs[i] = mcs_map
#                 filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
#                 filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
#                 img1 = filtered_im1
#                 img2 = filtered_im2
#             #msssim = torch.stack(msssim, dim=0)
#             #mcs = torch.stack(mcs, dim=0)
#             value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*
#                                     (msssim[levels-1]**weight[levels-1]))
#             return value
#         def forward(self, img1, img2):
#             print("Debug")
#             return self.ms_ssim(img1, img2)
