# -*- coding:utf-8 -*-
""" Build your own dataset DogCat.
__init__(), __getitem__(), __len()__ 
these three methods are necessarily defined.
set data file directory, split into training (needs Backprop), validation and test set
and define various transformation of images for different set.
"""

import os
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T


class DogCat(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True,test=False):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        root = data/train or data/test, dataset location directory
        '''
        # no super(DogCat, self).__init__() method, no inheritage form data.Dataset

        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)  
        # returns a list containing elements like below samples,
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        
        # sort the list using if statement because of differences in test and training set 
        # sorted(list, key=lambda x:x[dimension index]) Python built-in function
        # sort the element x in list according to x's specified dimension
        if self.test:
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2]))  # in digital order
            
        imgs_num = len(imgs)
        
        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)
        
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[ : int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num) : ]
            # training : validation = 7 : 3
    
        if transforms is None:

            # data conversion
            # training, validation and test data differs
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])

            if self.test or not train: 
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                    ]) 
            else :
                # training data transformation including data augmentation by default
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])
        else:
            self.transforms = transforms
        # transforms is used in methoed __getitem__()
                
        
    def __getitem__(self,index):
        '''
        一次返回一张图片的数据 (data, label)
        '''
        img_path = self.imgs[index]  # imgs is a list containing all data
        if self.test: label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        # set digit of image name as label for test set
        else: label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # set 1 or 0 as label for training and validation set
        data = Image.open(img_path)  # PIL format
        data = self.transforms(data)    # Tensor format
        return data, label
    
    def __len__(self):
        return len(self.imgs)
