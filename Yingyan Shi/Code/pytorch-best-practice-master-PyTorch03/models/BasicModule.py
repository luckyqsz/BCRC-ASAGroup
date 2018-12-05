# -*- coding:utf-8 -*-
""" In practice, directly call model.save(save_path) and model.load(opt.load_path).
In general, other models defined by yourself should inherit this BasicModule.
"""

import torch as t
import time


class BasicModule(t.nn.Module):
    '''
    封装了nn.Module, 主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self)) # 默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))
        # torch.load(model_path) returns a OrderedDict
        # load an object saved with torch.save() from a file
        # First load, then assign.

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        For example, AlexNet_0710_23:57:29.pth
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        # torch.save(network.cpu().state_dict(), 't.pth'), torch.load('t.pth')
        return name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
