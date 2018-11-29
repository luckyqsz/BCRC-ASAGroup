# Source: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorial/03-advanced/generative_adversarial_network/
# Tasteful coding style worth learning and imitating.
# PyTorch pros&cons
# Pros:
# Built-in data loading and augmentation, very nice!
# Training is fast, maybe even a little bit faster.
# Very memory efficient!
# Cons:
# No progress bar, sad :(
# No built-in log.


import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])  
                                     # Normalized_image = (image - mean) / std, where mean and std are specified manually 
                                     # instead of computing from raw data

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())  # Output a probability value

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer, i.e. adaptive moment estimation (pinyin: shiyingxing jvguji)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)	# The output of G(z) in the range (-1, 1), so change it in the range (0, 1).

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)   # the number of batches
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):   
        images = images.reshape(batch_size, -1).to(device)  # images is a tensor in the range [0.0, 1.0]
        # for i, data in enumerate(train_loader):
        # inputs, labels = data
        # load samples from data_loader with the number of batchsize once

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Back propagation and optimize
        d_loss = d_loss_real + d_loss_fake  # Positive and negative samples are equal in a batch during training the classifier.
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        # In practice, log(1-D(G(z))) may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence
        # they are clearly different from the training data. In this case, log(1-D(G(z))) saturates. Rather than training G to minimize log(1-D(G(z))) we can train G to 
        # maximize logD(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning.
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)  # images.size(0) is the value of batchsize, images.reshape(batchsize, channels, x, y)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')  # Returns a dictionary containing a whole state of the module. Both parameters and persistent buffers are included.
torch.save(D.state_dict(), 'D.ckpt')  # Keys are corresponding parameter and buffer names.

# Example:
# >>> module.state_dict().keys()
# ['bias', 'weight']
#
# ======================================
# PyTorch Documentation
# 
# class torch.device
# A torch.device is an object representaing the device on which a toch.Tensor is or will be allocated.
# The torch.device contains a device type ('cpu' or 'cuda') and optional device ordinal for the device type.
# If the device ordinal is not present, this represents the current device for the device type; e.g. 
# a torch.Tensor constructed with device 'cuda' is equivalent to 'cuda:x' where X is the result of 
# torch.cuda.current_device()
#
#
# ======================================
# module torchvision.transforms
# Transforms are common image transforms. They can be chained together using Compose.
# class torchvision.transforms.Compose(transforms)
# Composes several transforms together.
# Parameters: transforms(list of Transform object) 
#
# 1. Tranforms on PIL Image
#   class torchvision.transforms.
#   CenterCrop(size)    Crops the given PIL Image at the center.
#   ColorJitter(Brightness=0, contrast=0, saturation=0, hue=0)  Randomly change the brightness, contrast and saturation of an image.
#   FiveCrop(size)  Crop the given PIL Image into four corners and the central crop.
#   Grayscale(num_out_channels=1) Convert image to grayscale. If num_output_channels == 3: returned image is 3 channel with R == G == B.
#   LinearTransformation(transfromation_matrix) Transform a tensor image with a square transformation matrix computed offline.
#   Pad(padding, fill=0, padding_mode='constant')   Pad the given PIL Image on all sides with the given "pad" value.
#   RandomAffine(degrees, translate=None. scale=None, shear=None, resample=False, fillcolor=0)
#   RandomApply(transforms, p=0.5)
#   RandomChoice(transforms)
#   RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
#   RandomGrayscale(p=0.1)
#   RandomHorizontalFlip(p=0.5)
#   RandomOrder(transforms)
#   RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33333333333333), interpolation=2)
#   RandomRotation(degrees, resample=False, expand=False, center=None)
#   RandomSizedCrop()
#   RandomVerticalFlip(p=0.5)
#   Resize(size, interpolation=2)
#   Scale()
#   TenCrop()
#
# 2. Tranforms on torch.*Tensor
#   class torchvision.transforms.Norlize(mean, std)   Normalize a tensor image with mean and standard deviation. Given mean:(M1, ..., Mn) and
#   std:(S1, ..., Sn) for n channels, this transform will normalize each channel of the input torch.*Tensor, i.e. 
#   input[channel] = (input[channel] - mean[channel]) / std[channel]
#   This transform acts in-place, i.e. it mutates the input tensor.
#
# 3. Conversion Transforms
#   ToPILImage()
#   ToTensor()    Convert a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
# 
# 4. Generic Transforms
#
# 5. Functional Transforms
#
# =================================================
# torchvision.datasets
# All datasets are subclasses of torch.utils.data.Dataset, i.e., they have __getitem__ and __len__ methods implemented. Hence, they can all be passed to a
# torch.utils.data.DataLoader which can load multiple samples parallelly using torch.multiprocessing workers.
#
# class torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
#
# ===================================================
#
# Save and load the entire model
# torch.save(state, dir), dir repesents absolute directory and file name, i.e., '/home/yingyan/model.pkl'
# torch.save(model_object, 'model.pkl')
# model = torch.load('model.pkl')
# 
# Only save and load model parameters (recommended)
# torch.save(model_object.state_dict(), 'params.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))
#