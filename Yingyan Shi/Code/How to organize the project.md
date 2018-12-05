Yingyan Shi

shiyingyan12@126.com

November 2018

Brain Chip Research Center, Fudan University

------

[TOC]

# How to organize the project?

Question: 

- How to organize code to have good readability and extensibility and be more pythonic ?

* Pursuit of being equipped with well file organization and clean code detail.
* Especially PyTorch deep learning framework for computer vision.

When we perform experiments or projects related to Deep Learning, a great deal of modification and trial would be applied, aiming at getting the optimal results. Generally, program is supposed to conclude those major functions below:

1. model definition
2. data process and load
3. model training (train & validate)
4. visualization of training progress
5. test/inference

Some additional structural requirements are expected to meet as well:

1. model is highly reconfigurable, modifying parameters and  architecture easily
2. project is well organized and understandable
3. concise annotation is strongly advised

## 1. Project organization

```tx
├── checkpoints/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── get_data.sh
├── models/
│   ├── __init__.py
│   ├── AlexNet.py
│   ├── BasicModule.py
│   └── ResNet34.py
└── utils/
│   ├── __init__.py
│   └── visualize.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
```

* checkpoints/: save the training or trained model, so that model could be loaded again to resume training in case of the program breaking down.
* data/: operation involving data, including data preprocessing and dataset building.
* models/: model definition, many different models exist here (i.e. VGG16 and ResNet34), one model corresponding one file.
* utils/: utility functions used in the main function, i.e. visualization function.
* config.py: configuration file, containing all configurable variables and their respective default values.
* main.py: main file, the entry to training and testing program, using various commands to specify different operations and arguments.
* requirements.txt: dependent 3rd-party libraries or packages.
* README.md: the necessary introduction to this program.

## 2. About \_\_init_\_.py

Almost every folder has a file named \_\_init\_\_\.py . One folder would be a Python package if it contains \_\_init_\_.py . Note that \_\_init\_\_.py could be empty file or define attributes and methods of this package.

Other programs are able to import corresponding modules or functions from this package only when \_\_init_\_.py exists.

## 3. Data loading

Steps:

1. class **Dataset** is used to pack your own database.
2. class **Dataloader** is used to achieve data loading parallelly.

For training set, we hope to perform data augmentation technique, i.e. random crop, random flip and adding noise, while validation set and test set do not need.

For this program, Dogs vs Cats is a conventional binary classification, of  which the training set contains 25,000 images located in the same directory named in the format `<category>.<number>.jpg`, i.e. `cat.10000.jpg`, `dog.100.jpg`, and test set contains 12,500 images named in the format `<number>.jpg`, i.e. `1000.jpg`. Users train their models in the given training set and test in the given testing set, outputting the probability of object in the image being a dog. Finally, a CSV file recording test results is submitted.

```python
# dataset.py

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
 
class DogCat(data.Dataset):
   
   def __init__(self, root, transforms=None, train=True, test=False):
       """
       Goal: acquire the directory of all images, and split data into training, validation and test set.
       """
       self.test = test
       imgs = [os.path.join(root, img) for img in os.listdir(root)]
       # ============================ sort imgs list =============================
       # The file name is different between training set and validatin set
       # imgs contains:
       # test1: data/test1/8973.jpg
       # train: data/train/cat.10004.jpg 
       if self.test:
           imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        	# x.split('.') returns the split list containing string, then convert str into int
       else:
           imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))  # in digital order
       		# sorted(), Python built-in function
            # sorted(list, key=lambda x:x[dimension index])
            # sort the element x in list according to x's specified dimension
        
       imgs_num = len(imgs)
       
       # Split into traing set and validation set, 7:3
       if self.test:
           self.imgs = imgs
       elif train:
           self.imgs = imgs[:int(0.7*imgs_num)]
       else :
           self.imgs = imgs[int(0.7*imgs_num):]            
   
       if transforms is None:
       
           # Data conversion, data between training and validation differs           
           normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])
           # mean and std here is computed over images in ImageNet, leading a better performance probably
 
           # Data augmentation not applied to validation and test set
           if self.test or not train: 
               self.transforms = T.Compose([
                   T.Scale(224),
                   T.CenterCrop(224),
                   T.ToTensor(),
                   normalize
               ]) 
           
        # training set needs data augmentation
           else :
               self.transforms = T.Compose([
                   T.Scale(256),
                   T.RandomSizedCrop(224),
                   T.RandomHorizontalFlip(),
                   T.ToTensor(),
                   normalize
               ]) 
               
       
   def __getitem__(self, index):
       """
       return data of a image
       for test set without label, return image ID, i.e. 1000.jpg returns 1000
       """
       img_path = self.imgs[index]
       if self.test: 
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
       else: 
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
       data = Image.open(img_path)
       data = self.transforms(data)
       return data, label
   
   def __len__(self):
       """
       return the number of all images in dataset
       """
       return len(self.imgs)
```

Put time-consuming operations like file reading into  `__getitem__` function, and use multi-process to accelerate running time. Loading all images into memory will not only consume much time but also occupy large memory. Hence, we use Dataloader to load images by batch.

```python
train_dataset = DogCat(opt.train_data_root, train=True)
trainloader = DataLoader(train_dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers)
                  
for ii, (data, label) in enumerate(trainloader):
	train()
```

## 4. Model definition

Model definition is saved in the directory `models/`, BasicModule is the easy package for nn.Module providing interface to quick loading and model save.

```python
class BasicModule(t.nn.Module):
   """
   Pack nn.Module simply, and provide two method: save and load
   """
 
   def __init__(self, opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self))  # model's default name
 
   def load(self, path):
       """
       load model located in the specified path
       """
       self.load_state_dict(t.load(path))
        # torch.load(model_path) returns a OrderedDict
    	# loads an object saved with torch.save() from a file
 
   def save(self, name=None):
       """
       save model, named after "model name + time" by default
       i.e. AlexNet_0710_23:57:29.pth
       """
       if name is None:
           prefix = 'checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            # format the date, time.strftime(format[, tupletime])
            # receive time tuple, and return readable string representing localtime
            # %y (00-99), %m (01-12), %d (0-31)
            # H% (0-23), %l (01-12), %M (00-59), %S(00-59)
       t.save(self.state_dict(), name)  
       # torch.save(network.cpu().state_dict(), 't.pth'), torch.load('t.pth')
       return name
```

In practice, directly call `model.save()` and `model.load(opt.load_path)`.

Other models defined by yourself would inherit this BasicModule, then build your own specific model. AlexNet.py implement AlexNet, ReNet34.py implement ResNet34. In `models/__init__.py`, 

```python
from .AlexNet import AlexNet
from .ResNet import ResNet34
```

Then in the main function, we can use in this way:

```python
from models import AlexNet
```

or

```python
import models
model = models.AlexNet()
```

or

```python
import models
model = getattr(models, 'AlexNet')()
```

The last usage is critical. The newly added model is just add in  `models/__init__.py` it this way

```python
from .new_module import NewModule
```

Tips about model definition:

1. nn.Sequential is preferred (i.e. AlexNet)
2. Pack often used structure into submodule (i.e. Inception in GoogLeNet, Residual Block in ResNet)
3. Use function to generate repetitive and regular structure (i.e. many variants of VGG or ResNet are build by many similar blocks)

## 5. Tool function

All helper functions should be implemented in the directory `utils/`, imported when necessary.

```python
# -*- coding:utf-8 -*-
import visdom
import time
import numpy as np
 
class Visualizer(object):
   '''
   封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
   或者`self.function`调用原生的visdom接口
   比如 
   self.text('hello visdom')
   self.histogram(t.randn(1000))
   self.line(t.arange(0, 10),t.arange(1, 11))
   '''
 
   def __init__(self, env='default', **kwargs):
       self.vis = visdom.Visdom(env=env, **kwargs)
       
       # 画的第几个数，相当于横坐标
       # 比如（’loss',23） 即loss的第23个点
       self.index = {} 
       self.log_text = ''
   def reinit(self, env='default', **kwargs):
       '''
       修改visdom的配置
       '''
       self.vis = visdom.Visdom(env=env, **kwargs)
       return self
 
   def plot_many(self, d):
       '''
       一次plot多个
       @params d: dict (name, value) i.e. ('loss', 0.11)
       '''
       for k, v in d.iteritems():
           self.plot(k, v)
 
   def img_many(self, d):
       for k, v in d.iteritems():
           self.img(k, v)
 
   def plot(self, name, y, **kwargs):
       '''
       self.plot('loss', 1.00)
       '''
       x = self.index.get(name, 0)
       self.vis.line(Y=np.array([y]), X=np.array([x]),
                     win=unicode(name),
                     opts=dict(title=name),
                     update=None if x == 0 else 'append',
                     **kwargs
                     )
       self.index[name] = x + 1
 
   def img(self, name, img_, **kwargs):
       '''
       self.img('input_img', t.Tensor(64, 64))
       self.img('input_imgs', t.Tensor(3, 64, 64))
       self.img('input_imgs', t.Tensor(100, 1, 64, 64))
       self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
       '''
       self.vis.images(img_.cpu().numpy(),
                      win=unicode(name),
                      opts=dict(title=name),
                      **kwargs
                      )
 
   def log(self, info, win='log_text'):
       '''
       self.log({'loss':1, 'lr':0.0001})
       '''
 
       self.log_text += ('[{time}] {info} <br>'.format(
                           time=time.strftime('%m%d_%H%M%S'),\
                           info=info)) 
       self.vis.text(self.log_text, win)   
 
   def __getattr__(self, name):
       '''
       self.function 等价于self.vis.function
       自定义的plot,image,log,plot_many等除外
       '''
       return getattr(self.vis, name)
```

## 6. Configuration

There are so many variables in the phase of model definition, data processing and training which are provided default values and put in `config.py`.

```python
class DefaultConfig(object):
   env = 'default' # visdom 环境
   model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
   
   train_data_root = './data/train/' # 训练集存放路径
   test_data_root = './data/test1' # 测试集存放路径
   load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载
 
   batch_size = 128 # batch size
   use_gpu = True # use GPU or not
   num_workers = 4 # how many workers for loading data
   print_freq = 20 # print info every N batch
 
   debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
   result_file = 'result.csv'
     
   max_epoch = 10
   lr = 0.1 # initial learning rate
   lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
   weight_decay = 1e-4 # 损失函数
```

Configurable parameters include：

1. dataset parameters (file directory, batch_size)
2. training parameters (learning rate, epoch)
3. model parameters ()

As a result, we can use in main program this way:

```python
import models
from config import DefaultConfig
 
opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)
```

These are default parameters above, we can use the update function which update parameter configuration according dictionary.

```python
def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
            
        # 打印配置信息	
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k, getattr(self, k))
```

In this way, we can use terminal to set parameters to cover default configuration instead of modify config.py manually.

```python
opt = DefaultConfig()
new_config = {'lr':0.1,'use_gpu':False}
opt.parse(new_config)
opt.lr == 0.1
```

There is no function definition in config.py , so function 'help()' in main.py prints the source code in config.py  without printing parse function.

## 7. main.py

We can use an open-source command-line tool **fire** developed by Google to simply program execution.

When fire.Fire() is executed in program, we can use command-line arguments. That is to say,

```python
python file <function> [args,] {--kwargs}
```

Example:

```python
# example.py

import fire
def add(x, y):
 return x + y
 
def mul(**kwargs):
   a = kwargs['a']
   b = kwargs['b']
   return a * b
 
if __name__ == '__main__':
```

Then we can use this way,

```shell
python example.py add 1 2 # 执行add(1, 2)
python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2),kwargs={'a':1, 'b':2}
python example.py add --x=1 --y=2 # 执行add(x=1, y=2)
```

`main.py` code structure

```python
def train(**kwargs):
   """
   train
   """
   pass
    
def val(model, dataloader):
   """
   compute accuracy on validation set to assist training
   """
   pass
 
def test(**kwargs):
   """
   inference
   """
   pass
 
def help():
   """
   print help information
   """
   print('help')
 
if __name__=='__main__':
   import fire
   fire.Fire()
```

Consequently, we can use `python main.py <function> --args=xx` to start the progress of training or inference.

## 8. Training

Steps:

1. define deep neural networks
2. define data
3. define loss function and optimizer
4. compute metrics
5. start training
   1. training 
   2. visualize all metrics
   3. compute metrics in validation set

```python
def train(**kwargs):   
   # 根据命令行参数更新配置
   opt.parse(kwargs)
   vis = Visualizer(opt.env)
   
   # step1: 模型
   model = getattr(models, opt.model)()
   if opt.load_model_path:
       model.load(opt.load_model_path)
   if opt.use_gpu: model.cuda()
 
   # step2: 数据
   train_data = DogCat(opt.train_data_root,train=True)
   val_data = DogCat(opt.train_data_root,train=False)
   train_dataloader = DataLoader(train_data,opt.batch_size,
                       shuffle=True,
                       num_workers=opt.num_workers)
   val_dataloader = DataLoader(val_data,opt.batch_size,
                       shuffle=False,
                       num_workers=opt.num_workers)
   
   # step3: 目标函数和优化器
   criterion = t.nn.CrossEntropyLoss()
   lr = opt.lr
   optimizer = t.optim.Adam(model.parameters(),
                           lr = lr,
                           weight_decay = opt.weight_decay)
       
   # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
   loss_meter = meter.AverageValueMeter()
   confusion_matrix = meter.ConfusionMeter(2)
   previous_loss = 1e100
 
   # 训练
   for epoch in range(opt.max_epoch):
       
       loss_meter.reset()
       confusion_matrix.reset()
 
       for ii,(data,label) in enumerate(train_dataloader):
 
           # 训练模型
           input = Variable(data)
           target = Variable(label)
           if opt.use_gpu:
               input = input.cuda()
               target = target.cuda()
           optimizer.zero_grad()
           score = model(input)
           loss = criterion(score,target)
           loss.backward()
           optimizer.step()
           
           # 更新统计指标以及可视化
           loss_meter.add(loss.data[0])
           confusion_matrix.add(score.data, target.data)
 
           if ii%opt.print_freq==opt.print_freq-1:
               vis.plot('loss', loss_meter.value()[0])
               
               # 如果需要的话，进入debug模式
               if os.path.exists(opt.debug_file):
                   import ipdb;
                   ipdb.set_trace()
 
       model.save()
 
       # 计算验证集上的指标及可视化
       val_cm,val_accuracy = val(model,val_dataloader)
       vis.plot('val_accuracy',val_accuracy)
       vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
       .format(
                   epoch = epoch,
                   loss = loss_meter.value()[0],
                   val_cm = str(val_cm.value()),
                   train_cm=str(confusion_matrix.value()),
                   lr=lr))
       
       # 如果损失不再下降，则降低学习率
       if loss_meter.value()[0] > previous_loss:          
           lr = lr * opt.lr_decay
           for param_group in optimizer.param_groups:
               param_group['lr'] = lr
               
       previous_loss = loss_meter.value()[0]
```

## 9. Validation

Set the model in evaluation mode (`model.eval()`). Do not forget to set the model in training mode after validation (`model.training()`). Those two line codes will influence the running mechanism of some certain layers, i.e. BatchNorm and Dropout.

```python
def val(model,dataloader):
   '''
   计算模型在验证集上的准确率等信息
   '''
   # 把模型设为验证模式
   model.eval()
   
   confusion_matrix = meter.ConfusionMeter(2)
   for ii, data in enumerate(dataloader):
       input, label = data
       val_input = Variable(input, volatile=True)
       val_label = Variable(label.long(), volatile=True)
       if opt.use_gpu:
           val_input = val_input.cuda()
           val_label = val_label.cuda()
       score = model(val_input)
       confusion_matrix.add(score.data.squeeze(), label.long())
 
   # 把模型恢复为训练模式
   model.train()
   
   cm_value = confusion_matrix.value()
   accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /\
                (cm_value.sum())
   return confusion_matrix, accuracy
```

## 10. Test

When in the phase of testing,  the probability of object in each image belonging to the dog class is computed and saved as CSV file.

```python
def test(**kwargs):
  opt.parse(kwargs)
  # 模型
   model = getattr(models, opt.model)().eval()
   if opt.load_model_path:
       model.load(opt.load_model_path)
   if opt.use_gpu: model.cuda()
 
   # 数据
   train_data = DogCat(opt.test_data_root,test=True)
   test_dataloader = DataLoader(train_data,\
                               batch_size=opt.batch_size,\
                               shuffle=False,\
                               num_workers=opt.num_workers)
   
   results = []
   for ii,(data,path) in enumerate(test_dataloader):
       input = t.autograd.Variable(data,volatile = True)
       if opt.use_gpu: input = input.cuda()
       score = model(input)
       probability = t.nn.functional.softmax\
           (score)[:,1].data.tolist()      
       batch_results = [(path_,probability_) \
           for path_,probability_ in zip(path,probability) ]
       results += batch_results
   write_csv(results,opt.result_file)
   return results
```

## 11. Help

In order to get this program used easily, a help function is highly recommended to provide introducing how to use this program. Hera we use method ''inspect" in Python standard library to automatically acquire config source code.

```python
def help():
  '''
  打印帮助的信息： python file.py help
   '''
   
   print('''
   usage : python {0} <function> [--args=value,]
   <function> := train | test | help
   example: 
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} help
   avaiable args:'''.format(__file__))
 
   from inspect import getsource
   source = (getsource(opt.__class__))
   print(source)
```

When user executes `python main.py help`, information below will be printed.

```bash
usage : python main.py <function> [--args=value,]
  <function> := train | test | help
  example: 
           python main.py train --env='env0701' --lr=0.01
           python main.py test --dataset='path/to/dataset/'
           python main.py help
   avaiable args:
class DefaultConfig(object):
   env = 'default' # visdom 环境
   model = 'AlexNet' # 使用的模型
   
   train_data_root = './data/train/' # 训练集存放路径
   test_data_root = './data/test1' # 测试集存放路径
   load_model_path = 'checkpoints/model.pth' # 加载预训练的模型

   batch_size = 128 # batch size
   use_gpu = True # user GPU or not
   num_workers = 4 # how many workers for loading data
   print_freq = 20 # print info every N batch

   debug_file = '/tmp/debug' 
   result_file = 'result.csv' # 结果文件
     
   max_epoch = 10
   lr = 0.1 # initial learning rate
   lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
   weight_decay = 1e-4 # 损失函数
```

## 12. Execution

In terminal, we can enter such instructions like those:

```shell
# 训练模型
python main.py train 
        --train-data-root=data/train/ 
        --load-model-path='checkpoints/resnet34_16:53:00.pth' 
        --lr=0.005 
        --batch-size=32 
        --model='ResNet34'  
        --max-epoch = 20
 
# 测试模型
python main.py test
       --test-data-root=data/test1 
       --load-model-path='checkpoints/resnet34_00:23:05.pth' 
       --batch-size=128 
       --model='ResNet34' 
       --num-workers=12
 
# 打印帮助信息
python main.py help
```

## 13. Appendix

1. Compared with `config.py`, Python standard library `argparse` is less convenient or straightforward with more codes. Every time you want to add a command line argument, below code should be added into your program: 

   `parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')`

2. Someone likes to integrate training progress with model definition, for example,

   ```python
   class MyModel(nn.Module):
      def __init__(self,opt):
            self.dataloader = Dataloader(opt)
            self.optimizer  = optim.Adam(self.parameters(),lr=0.001)
            self.lr = opt.lr
            self.model = make_model()
        
        def forward(self,input):
            pass
        
        def train_(self):
            # 训练模型
            for epoch in range(opt.max_epoch)
              for ii,data in enumerate(self.dataloader):
                  self.train_step(data)
              model.save()
      
        def train_step(self):
            pass
   ```

   Or specially designs a object Trainer,

   ```python
   import heapq
   from torch.autograd import Variable
    
   class Trainer(object):
    
        def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.dataset = dataset
            self.iterations = 0
    
        def run(self, epochs=1):
            for i in range(1, epochs + 1):
                self.train()
    
        def train(self):
            for i, data in enumerate(self.dataset, self.iterations + 1):
                batch_input, batch_target = data
                self.call_plugins('batch', i, batch_input, batch_target)
                input_var = Variable(batch_input)
                target_var = Variable(batch_target)
      
                plugin_data = [None, None]
      
                def closure():
                    batch_output = self.model(input_var)
                    loss = self.criterion(batch_output, target_var)
                    loss.backward()
                    if plugin_data[0] is None:
                        plugin_data[0] = batch_output.data
                        plugin_data[1] = loss.data
                    return loss
              self.optimizer.zero_grad()
              self.optimizer.step(closure)
      
            self.iterations += i
   ```

## 14. requirements.txt

In project implemented by Python, there has to be a file *requirements.txt* to record all dependent packages and respective version, in order to deploy this project conveniently in new environment.

1. use pip to generate in original virtual environment

   `(venv) $ pip freeze >requirements.txt`

2. build copy of this original virtual environment

   `(venv) $ pip install -r requirements.txt`

## 15. Reference

1. PyTorch Introduction https://blog.csdn.net/wzy_zju/article/details/78509662

2. chenyuntc/pytorch-best-practice:  A Guidance on PyTorch Coding Style Based on Kaggle Dogs vs. Cats 

   https://github.com/chenyuntc/pytorch-best-practice