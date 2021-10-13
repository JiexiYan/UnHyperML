from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from torchvision import transforms
from collections import defaultdict

from DataSet.CUB200 import MyData, default_loader, Generate_transform_Dict


class Stanford_Online_Products:
    def __init__(self, root=None, root_c=None, origin_width=256, width=227, ratio=0.16, transform=None,part_rate=0, noise_rate=0, HC=True):
        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = 'data/Cars196/'

        if root_c == '/home/yjx/CVPR2021/':
            train_txt = os.path.join(root_c, 'train.txt')
        else:           
            if part_rate == 0:
                if noise_rate == 0:
                    train_txt = os.path.join(root, 'train.txt')
                else:
                    train_txt = os.path.join(root, 'train_%.4f.txt'%noise_rate)    
            else: 
                if noise_rate == 0:
                    train_txt = os.path.join(root, 'train_part_%.4f.txt'%part_rate)
                else:
                    train_txt = os.path.join(root, 'train_part_%.4f_%.4f.txt'%(part_rate,noise_rate))
        # notification
        print('Notifaication: Using {} now!'.format(train_txt))
        test_txt = os.path.join(root, 'test.txt')
        
        if HC:
            self.train = MyData_HC(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        else:
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        
        
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'])


def testStanford_Online_Products():
    data = Stanford_Online_Products()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testStanford_Online_Products()


