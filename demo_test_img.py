# SR_Demo
import time
import cv2
import torch 
from torch.autograd import Variable
import pdb
import argparse
import numpy as np
import os 
import torch.backends.cudnn as cudnn
# from model import model
import scipy.io as scio  
cudnn.fastest = True

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')
parser.add_argument('--scale', type=int, default= 3, help='scale output size /input size')
parser.add_argument('--model', default='./model/model_new.pt', help='preTrained model')
parser.add_argument('--input', default='./data_test/wx_image_EH', help='input data path')
parser.add_argument('--output', default='./data_test/wx_image_EH', help='output data path')

parser.add_argument('--YCrCb',default=False, help='RGB to YCrCb ')
parser.add_argument('--Y',default=False, help='Y only ')
args = parser.parse_args()

def load_model(args):
    print('Loading model from {}...'.format(args.model))
    #  load model to cpu
    # myModel = torch.load(args.model, map_location=lambda storage, loc: storage)  
    myModel = torch.load(args.model)
    print(myModel)
    myModel.eval()
    myModel.cuda()
    return myModel

def sr_process(args, myModel):

    filelist = [f for f in os.listdir(args.input)]
    filelist.sort()
    w_ = 0
    h_ = 0
    time_sr = 0
    i = 0
    
    for f in filelist:

        print('The %dth frame : %s ...'%(i,f))
        frame = cv2.imread(os.path.join(args.input, f))
        
        w, h, c = frame.shape
        if w % 2 != 0:
            add_ = np.zeros([1, h, c])
            frame = np.row_stack((frame, add_))
            w += 1

        if h % 2 != 0:
            add_ = np.zeros([w, 1, c])
            frame = np.column_stack((frame, add_))
            h += 1

        w_ += w
        h_ += h
        frame_bicubic = cv2.resize(frame, (h*args.scale, w*args.scale), interpolation=cv2.INTER_CUBIC)
        if args.YCrCb:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        if args.Y:
            frame_YCrCb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
            w, h, c = frame.shape
            frame_YCrCb_x3 = cv2.resize(frame_YCrCb, (h*args.scale, w*args.scale), cv2.INTER_CUBIC)
            frame = np.zeros([w, h, 1])
            frame[:,:,0] = frame_YCrCb[:,:,0]
            w, h, c = frame.shape

        tensor_4D = torch.FloatTensor(1, c, w, h)
        t0 = time.time()
        tensor_4D[0,:,:,:] = torch.FloatTensor(frame.transpose(2,0,1))
        # pdb.set_trace()

        input = Variable(tensor_4D.cuda(), volatile=True)
        
        output = myModel(input)
        torch.cuda.synchronize()

        output_img = (output.data[0].cpu().numpy()).transpose(1,2,0)
        # output_img = output.data[0].numpy().transpose(1,2,0)
        time_sr = time_sr + (time.time() - t0) 

        print((time.time() - t0) )    


        output_img[output_img > 255] = 255
        output_img[output_img < 0] = 0
        out_sr = output_img.astype(np.uint8)
        # out_sr = cv2.resize(out_sr, (h*3, w*3), interpolation=cv2.INTER_CUBIC)
        if args.YCrCb:  
            out_sr = cv2.cvtColor(out_sr, cv2.COLOR_YCrCb2BGR)
        if args.Y:
            frame_YCrCb_x3[:,:,0] = out_sr[:,:,0]
            out_sr = cv2.cvtColor(frame_YCrCb_x3, cv2.COLOR_YCrCb2BGR)
        # out_sr = cv2.cvtColor(out_sr.astype(np.uint8), cv2.COLOR_BGR2YCrCb)[:,:,0]
        f = f[0:-4] + '_' + args.model.split('.')[-2].split('/')[-1].split('_',1)[-1] + '.png'
        # f = '/media/lzw/786435e1-97f2-40d5-a0ba-9fa657506869/sr/'+ \
        #     f[0:-4]  +  '_' + args.model.split('.')[-2].split('/')[-1].split('_',1)[-1] + '.png'
        cv2.imwrite(os.path.join(args.output, f), out_sr)
        # frame_bicubic = cv2.resize(out_sr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(f, frame_bicubic)
        i = i + 1        
    print('process all  sr time : {} / {} s '.format(time_sr, time_sr/ i ))

def main(args):

    myModel = load_model(args)
    sr_process(args, myModel)

if __name__ == "__main__":
    main(args)
