# -*- coding: utf-8 -*-
""" Training VGGNet with CIFAR10 """

    import torch 
    from torchvision import datasets,transforms
    import os
    import matplotlib.pyplot as plt
    import time
    #transform = transforms.Compose是把一系列图片操作组合起来，比如减去像素均值等。
    #DataLoader读入的数据类型是PIL.Image
    #这里对图片不做任何处理，仅仅是把PIL.Image转换为torch.FloatTensor，从而可以被pytorch计算
    transform = transforms.Compose(
        [
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ]
    )
    #训练集
    train_set = datasets.CIFAR10(root='drive/pytorch/Alexnet/', train=True, transform=transform, target_transform=None, download=True)
    #测试集
    test_set=datasets.CIFAR10(root='drive/pytorch/Alexnet/',train=False,download=True,transform=transform)
    trainloader=torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True,num_workers=0)
    testloader=torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=True,num_workers=0)
    classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    (data,label)=train_set[64]
    print(classes[label])

    X_example,y_example=next(iter(trainloader))
    print(X_example.shape)
    img=X_example.permute(0, 2, 3, 1)
    print(img.shape)
    import torchvision
    img=torchvision.utils.make_grid(X_example)
    img=img.numpy().transpose([1,2,0])
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


# 本次训练并不直接使用上面的VGGNet16，直接使用Pytorch的models里面预训练好的模型，
# 进行迁移学习，首先先下载模型，然后冻结前面的卷积层，仅对后面全连接层参数进行调整，
# 调整之后再进行迁移学习
    from torchvision import models

    vgg=models.vgg16(pretrained=True)
    
    for parma in vgg.parameters():
      parma.requires_grad = False
    
    vgg.classifier=torch.nn.Sequential(
        torch.nn.Linear(25088,4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096,4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096,10))
    
    cost=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(vgg.classifier.parameters(),lr=0.0001)
    
    import torch.optim as optim          #导入torch.potim模块
    import time
    from torch.autograd import Variable   # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
    import torch.nn as nn
    import torch.nn.functional as F
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
        y_pred = vgg(inputs)                # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
        _,pred=torch.max(y_pred.data,1)
        optimizer.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
        loss = cost(y_pred, labels)    # 计算损失值,criterion我们在第三步里面定义了
        loss.backward()                      # loss进行反向传播，下文详解
        optimizer.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
        running_loss += loss.item()       # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        running_corrects+=torch.sum(pred==labels.data)
        if(i % 2000 == 0):    # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
          print("Batch{},Train Loss:{:.4f},Train ACC:{:.4f}".format(i,running_loss/i,100*running_corrects/(32*i)))