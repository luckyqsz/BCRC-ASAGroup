#coding:utf8
from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block (BasicBlock)
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),  # padding = 1
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        # When dimension increases, shortcut is not None.
        # The variable name residual is not accurate, identity mapping actually.
        out += residual
        return F.relu(out)

class ResNet34(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        # input image size: (224, 224, 3)
        # root layer for all ResNet variants: RGB 3 inchannels, 7x7 conv, 64 outchannels, /2 stride, 
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),  
                # nn.Convv2d(inchannel, outchannel, kernel_size, stride, padding, bias)
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))  
                # nn.MaxPool2d(kernel_size, stride, padding)
        
        # 重复的layer，分别有3，4，6，3个residual block
        # _make_layer(inchannel, outchannel, block_num)
        self.layer1 = self._make_layer( 64, 128, 3)  # returns nn.Sequential()
        self.layer2 = self._make_layer( 128, 256, 4, stride=2)
        self.layer3 = self._make_layer( 256, 512, 6, stride=2)
        self.layer4 = self._make_layer( 512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        The dimensions of input and output of the first residual block of each layer differ,
        while of the rest residual block of each layer are the same. 
        '''
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
                # When the dimension of residual block's input and output not consistent,
                # shortcut path should be equipped with convolutional transformation with
                # kernel size 1x1.
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        # This first RedisualBlock of each layer increases dimensions, so shortcut is transformation path.
        # Other layers is identity mapping, so shortcut is N.
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)   # split list layers
        
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
