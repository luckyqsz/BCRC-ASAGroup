# -*- coding: utf-8 -*-
""" VGGNet implementation with PyTorch """

import torch
import torch.nn as nn
    
class VGGNet16(nn.Module):
    # 规整的网络, conv layer kernel size = 3, maxpool kernal size = 2
    def __init__(self, num_classes=10):
        super(VGGNet16,self).__init__()
        self.Conv=nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
                nn.Linear( 6 * 6 * 512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, num_classes),
        )
            
        def forward(self,inputs):
            x=self.Conv(inputs)
            x=x.view(-1,4*4*512)
            x=self.classifier(x)
            return x

vgg = VGGNet16()
print(vgg)

''' display the result
VGGNet16(
      (Conv): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU()
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU()
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU()
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU()
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU()
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU()
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=18432, out_features=1024, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5)
        (3): Linear(in_features=1024, out_features=1024, bias=True)
        (4): ReLU()
        (5): Dropout(p=0.5)
        (6): Linear(in_features=1024, out_features=1000, bias=True)
      )
    )
'''

# 可以使用全局平均池化层（GAP，Global Average Pooling）的方法代替全连接层。
# 全连接层将卷积层展开成向量之后不还是要针对每个feature map进行分类吗，
# GAP的思路就是将上述两个过程合二为一，一起做了。
# 它的思想是：用 feature map 直接表示属于某个类的 confidence map，
# 比如有4个类，就在最后输出4个 feature map，每个feature map中的值加起来求平均值，
# 这四个数字就是对应的概率或者叫置信度。然后把得到的这些平均值直接作为属于某个类别的 
# confidence value，再输入softmax中分类, 实验效果并不比用 FC 差，并且参数会少很多，
# 从而又可以避免过拟合。
# 这两者合二为一的过程我们可以探索到GAP的真正意义是:
# 对整个网路在结构上做正则化防止过拟合。
# 其直接剔除了全连接层中黑箱的特征，直接赋予了每个channel实际的类别意义。