# -*- coding: utf-8 -*-
""" GoogLeNet implementation with PyTorch. 
该网络引入Inception模块，Inception的目的是设计一种具有优良局部拓扑结构的网络，'
即对输入图像并行地执行多个卷积运算或池化操作，并将所有输出结果拼接为一个非常深的特征图。
因为 1*1、3*3 或 5*5 等不同的卷积运算与池化操作可以获得输入图像的不同信息，
并行处理这些运算并结合所有结果将获得更好的图像表征。
Source: https://mp.weixin.qq.com/s/s4WwmmKKi6655t5BIeGdsw
"""

import torch
from torchvision import datasets,transforms
import os
import matplotlib.pyplot as plt
import time

# transform = transforms.Compose是把一系列图片操作组合起来，比如减去像素均值等。
# DataLoader读入的数据类型是PIL.Image
# 这里对图片不做任何处理，仅仅是把PIL.Image转换为torch.FloatTensor，从而可以被pytorch计算
transform = transforms.Compose(
    [
        transforms.Scale([224,224]),
        transforms.ToTensor()        
        #transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ]
)
# 训练集
train_set = datasets.CIFAR10(root='drive/pytorch/inception/', train=True, transform=transform, target_transform=None, download=True)
# 测试集
test_set=datasets.CIFAR10(root='drive/pytorch/inception/',train=False,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True,num_workers=0)
testloader=torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=True,num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(data,label)=train_set[64]
print(classes[label])

''' 直接使用PyTorch的models里面预训练好的模型，进行迁移学习，
首先先下载模型，然后冻结所有的层，仅对后面全连接层参数进行调整
以及后面的AuxLogits和Mixed_7c两个模块的参数更新策略设置为
parma.requires_grad=True，允许更新参数，调整之后再进行迁移学习.
'''
from torchvision import models
inceptionv3=models.inception_v3(pretrained=True)

import torch
import torch.nn as nn
for parma in inceptionv3.parameters():
    parma.requires_grad = False
inceptionv3.fc=nn.Linear(in_features=2048, out_features=10, bias=True)
inceptionv3.AuxLogits.fc=nn.Linear(in_features=768, out_features=10, bias=True)
for parma in inceptionv3.AuxLogits.parameters():
    parma.requires_grad=True
for parma in inceptionv3.Mixed_7c.parameters():
    parma.requires_grad=True

# training model GoogLeNet
import torch.optim as optim          #导入torch.potim模块
import time
from torch.autograd import Variable   # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到import torch.nn as nnimport torch.nn.functional as F

optimizer=torch.optim.Adam(inceptionv3.parameters(),lr=0.0001)
epoch_n=5
for epoch in range(epoch_n):
  print("Epoch{}/{}".format(epoch,epoch_n-1))
  print("-"*10)
  running_loss = 0.0  #定义一个变量方便我们对loss进行输出
  running_corrects=0
  for i, data in enumerate(trainloader, 1): # 这里我们遇到了第一步中出现的trailoader，代码传入
    inputs, labels = data   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
    #inputs=inputs.permute(0, 2, 3, 1)
    #print("hahah",len(labels))
    y_pred = inceptionv3(inputs)                # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
    _,pred=torch.max(y_pred.data,1)
    optimizer.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
    loss = cost(y_pred, labels)    # 计算损失值,criterion我们在第三步里面定义了
    loss.backward()                      # loss进行反向传播，下文详解
    optimizer.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
    # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
    running_loss += loss.item()       # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
    running_corrects+=torch.sum(pred==labels.data)    
    if(i % 2 == 0):    # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
      print("Batch{},Train Loss:{:.4f},Train ACC:{:.4f}".format
      (i,running_loss/i,100*running_corrects/(32*i)))
