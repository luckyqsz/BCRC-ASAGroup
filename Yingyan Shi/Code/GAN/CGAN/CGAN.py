""" PyTorch-GAN/cgan.py at master · eriklindernoren/PyTorch-GAN  
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
Class label is fed into both G and D as input.
FC implementation.
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')  # 1 x 32 x 32
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)  # default 1 x 32 x 32

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):  # Multilayer perceptron
    def __init__(self):
        super(Generator, self).__init__()

        # an Embedding module containing op.n_classes tensors of size opt.n_classes
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # torch.nn..Embedding(num_embeddings, embedding_dim) a simple look-up table containing fixed dict
        # this module saves word embeddings and index them by subscripts.
        # num_embeddings(int)-the size of embedding dict
        # embedding_dim(int)-the size of embedding vector 
        # This self.label_emb contains initial embeddings whose weights are not learned yet.

        def block(in_feat, out_feat, normalize=True):  # fc layer component: BatchNorm and activation function 
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            # Batch normalization speeds up algorithmic convergence and facilitate algorithmic stability.
            # In order to keep the data distribution consistent, which would highly contribute to the generalization of model.
            # It would hamper the learning process of model if the data distribution of each epoch varies.
            # Batch normalization itself is one way of regularization which could replace Dropout layer.

        self.model = nn.Sequential(
            *block(opt.latent_dim+opt.n_classes, 128, normalize=False),  # conditional GAN: letent dim + class dim
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),  # symbol "*" means spliting the list into individual component
            nn.Linear(1024, int(np.prod(img_shape))),  # np.prod(tuple), return the product of array elements over a given axis.
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)  # along the last axis
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)  # split image shape tuple into individual elements
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # self.label_embedding is a look-up table whose input are indices and output are the content of LUT. 
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)  # first flatten the image then concatenate them.
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()  # Mean Square Error Loss, argument size_average=True by default.
auxiliary_loss = torch.nn.CrossEntropyLoss()  # not used :)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:  # transfer cpu to gpu
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # data type in GPU
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):  # n_row = n_col, batches_done = epoch * len(dataloader) + i
    """Saves a grid of generated digits ranging from 0 to n_classes"""  # number of classes for dataset
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)
    # from torchvision.utils import save_image

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):  # i represents step index

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))  # guarantee float data representation
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))  # (low, high, size)
        # random integer number of interval (0, n_classes) of size batch_size

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)  # average loss of batch by default

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

''' 为什么要使用Batch Norm呢？上边所说的加快训练速度只是一个简单的原因，在简单的深层网络中，如果前层中的参数改变，
后层中的参数也会跟着变化，如果加上Batch Norm，即使输入数据的分布会有变化，但是他们的均值方差可控，从而使变化带来的影响减小，
使各个层之间更加独立，更利于每层‘专门做自己的事情’。
具体运算：对于一个batch中的属于同一个channel的所有元素进行求均值与方差。
batch normalization层能够学习到的参数，对于一个特定的channel而言实际上是两个参数，gamma与beta，
对于total的channel而言实际上是channel数目的两倍。
Pytorch中的Batch Normalization操作  http://www.mamicode.com/info-detail-2378483.html
--------------------- 
作者：加勒比海鲜王 
来源：CSDN 
原文：https://blog.csdn.net/yinruiyang94/article/details/78002342 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
