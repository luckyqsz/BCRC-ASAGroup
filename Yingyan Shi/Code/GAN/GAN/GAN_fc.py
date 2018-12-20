""" Implement original GAN in the form of MLP using MNIST dataset with PyTorch.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
 
def showimg(images,count):  # images torch.Tensor
    images=images.detach().numpy()[0:16,:]  # batchsize = 100
    images=255*(0.5*images+0.5)  # 
    images = images.astype(np.uint8)
    grid_length=int(np.ceil(np.sqrt(images.shape[0])))  # grid_length = 4 (sqrt(16))
    plt.figure(figsize=(4,4))
    width = int(np.sqrt((images.shape[1])))  # width = 28 (sqrt(28x28))
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0)
    # gs.update(wspace=0, hspace=0)
    print('starting...')
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([width,width]),cmap = plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    print('showing...')
    plt.tight_layout()
    plt.savefig('./GAN_Image/%d.png'%count, bbox_inches='tight')
 
def loadMNIST(batch_size):  #MNIST图片的大小是28*28
    trans_img=transforms.Compose([transforms.ToTensor()]) # value range: [0, 1]
    trainset=MNIST('./data',train=True,transform=trans_img,download=True)
    testset=MNIST('./data',train=False,transform=trans_img,download=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset,testset,trainloader,testloader
 
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(784,300),
            nn.LeakyReLU(0.2),
            nn.Linear(300,150),
            nn.LeakyReLU(0.2),
            nn.Linear(150,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.dis(x)
        return x
 
class generator(nn.Module):
    def __init__(self,input_size):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(input_size,150),
            nn.ReLU(True),
            nn.Linear(150,300),
            nn.ReLU(True),
            nn.Linear(300,784),
            nn.Tanh()
        )
    def forward(self, x):
        x=self.gen(x)
        return x
 
if __name__=="__main__":
    criterion=nn.BCELoss()  # Binary Cross Entropy
    num_img=100  # batchsize
    z_dimension=100  # latent space dimension
    D=discriminator()
    G=generator(z_dimension)
    trainset, testset, trainloader, testloader = loadMNIST(num_img)  # data
    d_optimizer=optim.Adam(D.parameters(),lr=0.0003)
    g_optimizer=optim.Adam(G.parameters(),lr=0.0003)
    '''
    交替训练的方式训练网络
    先训练判别器网络D再训练生成器网络G
    不同网络的训练次数是超参数
    也可以两个网络训练相同的次数
    这样就可以不用分别训练两个网络
    '''
    count=0
    #鉴别器D的训练,固定G的参数
    epoch = 100
    gepoch = 1
    for i in range(epoch):
        for (img, label) in trainloader:
            # num_img=img.size()[0]
            real_img=img.view(num_img,-1) #展开为28*28=784
            real_label=torch.ones(num_img) #真实label为1
            fake_label=torch.zeros(num_img) #假的label为0
 
            #compute loss of real_img
            real_out=D(real_img) #真实图片送入判别器D输出0~1
            d_loss_real=criterion(real_out,real_label)#得到loss
            real_scores=real_out#真实图片放入判别器输出越接近1越好
 
            #compute loss of fake_img
            z=torch.randn(num_img,z_dimension)#随机生成向量
            fake_img=G(z)#将向量放入生成网络G生成一张图片
            fake_out=D(fake_img)#判别器判断假的图片
            d_loss_fake=criterion(fake_out,fake_label)#假的图片的loss
            fake_scores=fake_out#假的图片放入判别器输出越接近0越好
 
            #D bp and optimize
            d_loss=d_loss_real+d_loss_fake
            d_optimizer.zero_grad() #判别器D的梯度归零
            d_loss.backward() #反向传播
            d_optimizer.step() #更新判别器D参数
 
            #生成器G的训练compute loss of fake_img
            for j in range(gepoch):
                fake_label = torch.ones(num_img)  # 真实label为1, trick explained in GAN paper.
                z = torch.randn(num_img, z_dimension)  # 随机生成向量
                fake_img = G(z)  # 将向量放入生成网络G生成一张图片
                output = D(fake_img)  # 经过判别器得到结果
                g_loss = criterion(output, fake_label)#得到假的图片与真实标签的loss
                #bp and optimize
                g_optimizer.zero_grad() #生成器G的梯度归零, Actually D's gradients are accumulated.
                g_loss.backward() #反向传播, PyTorch define the computational graph dynamically.
                g_optimizer.step() #更新生成器G参数, only g_optimizer works, d_optimizer does not.
        
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                i, epoch, d_loss.data[0], g_loss.data[0], # access number in Tensor using .data[0]
                real_scores.data.mean(), fake_scores.data.mean()))
        showimg(fake_img,count) # every epoch
        # plt.show()
        count += 1
