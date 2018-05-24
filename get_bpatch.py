import cv2
import os
import sys
import numpy as np
import pdb



data_path = './data'
data_resize = '8k.4k_LR_x3'
data_no_compress = 'no_compress'
data_re_compress = 'recompress_'
data_re_compress_lable = 'recompress_label_'
data_compress = 'compress'

filelist = [f for f in os.listdir(os.path.join(data_path, data_compress))]
i = 0
for f in filelist:
    i = i +1
    print('%s ...'%(f))
    im_compress = cv2.imread(os.path.join(data_path, data_compress, f))
    f = f[0:-4]+'.png'
    im_no_compress = cv2.imread(os.path.join(data_path, data_no_compress, f))


    # w, h, c = im_no_compress.shape
    # if w % 2 != 0:
    #     add_ = np.zeros([1, h, c])
    #     im_no_compress = np.row_stack((im_no_compress, add_))
    #     w += 1

    # if h % 2 != 0:
    #     add_ = np.zeros([w, 1, c])
    #     im_no_compress = np.column_stack((im_no_compress, add_))
    #     h += 1



    # im_resize = cv2.resize(im_no_compress, (h//3, w//3), cv2.INTER_CUBIC)

    # cv2.imwrite(os.path.join(data_path, data_resize, f), im_resize) 





    # for n in im_type:
    #     f_name = f[0:-4] + n
    #     if os.path.exists((os.path.join(data_path, data_no_compress, f_name))):
    #         im_no_compress = cv2.imread(os.path.join(data_path, data_no_compress, f_name))
    # im_resize = cv2.resize(im_no_compress, (h//3, w//3), cv2.INTER_LINEAR)

    f = f[0:-4] + '_.png'
    # cv2.imwrite(os.path.join(data_path, data_re_compress_lable, f), im_no_compress[12:,12:,:]) 
    cv2.imwrite(os.path.join(data_path, data_re_compress, f), im_compress[4:,4:,:]) 

    '''

    # if w > h:
    #     w_scale = w // 1920
    #     h_scale = h // 1080
    #     for m in range(w_scale):
    #         for n in range(h_scale):
    #             name = '{:0>4}'.format(num)+ '.png'
    #             print(name)
    #             img = im_no_compress[1920*m : 1920*(m+1), 1080*n : 1080*(n+1), :]
    #             cv2.imwrite(os.path.join(data_path, data_patch, name), img) 
    #             num += 1
    # else:
    #     h_scale = h // 1920
    #     w_scale = w // 1080
    #     for m in range(w_scale):
    #         for n in range(h_scale):
    #             name = '{:0>4}'.format(num) + '.png'
    #             print(name)
    #             img = im_no_compress[1080*m : 1080*(m+1), 1920*n : 1920*(n+1), :]
    #             cv2.imwrite(os.path.join(data_path, data_patch, name), img) 
    #             num += 1
    # i += 1
    '''