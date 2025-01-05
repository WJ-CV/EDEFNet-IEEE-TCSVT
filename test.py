import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from options import opt
from torch import nn
from Encoder import Mnet
from Decoder import Main_Decoder, Aux_decoders
from data import test_dataset
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Encoder_net = nn.DataParallel(Mnet()).cuda()
Main_Decoder = nn.DataParallel(Main_Decoder()).cuda()
data_path = '/'
Encoder_net.eval()
Main_Decoder.eval()
fps = 0
if __name__ == '__main__':
    for j in ['MAE']:  #,'MAE'
        test_datasets = ['SIP', 'NJUD','DES','NLPR','STERE','DUT','LFSD','SSD',] #
        for dataset in test_datasets:
            time_s = time.time()
            E_model_path = os.path.join('./model/', 'Best_' + str(j) + '_E_epoch.pth') 
            D_model_path = os.path.join('./model/', 'Best_' + str(j) + '_D_epoch.pth') 
            Encoder_net.load_state_dict(torch.load(E_model_path))
            Main_Decoder.load_state_dict(torch.load(D_model_path))

            sal_save_path = os.path.join('./output/', dataset + '-' + str(j) + '/')
            if not os.path.exists(sal_save_path): os.makedirs(sal_save_path)

            gt_root = data_path + dataset
            root_ptah = gt_root
            test_loader = test_dataset(gt_root, root_ptah, opt.trainsize)
            nums = test_loader.size
            for i in range(test_loader.size):
                image, gt, t, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                t = t.cuda()
                sup_body, sup_edge, sup_all = Encoder_net(image, t)
                score_body, score_edge, score_1, score = Main_Decoder(sup_body, sup_edge, sup_all)
                res = F.upsample(score, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                print('save img to: ', sal_save_path + name)
                cv2.imwrite(sal_save_path + name, res * 255)
            time_e = time.time()
            fps += (nums / (time_e - time_s))
            print("FPS:%f" % (nums / (time_e - time_s)))
            print('Test Done!')
        print("Total FPS %f" % fps) # this result include I/O cost

