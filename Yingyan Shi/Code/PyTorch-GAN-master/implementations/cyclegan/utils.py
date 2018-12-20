import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

# Update the discriminators using a history of generated images rather than the ones produced
# by the latest generations. Keep an image buffer that stores the 50 previously created images.
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  # data is (batch size, 3, h, w)
    # push data (batchsize), pop data

        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)  # add one dim
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)  # self.data update randomly
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)  # self.data no update
        return Variable(torch.cat(to_return))  # this is why we use torch.unsqueeze() before.
        # torch.cat(sequence of Tensors, dim=0) -> Tensor

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs  # number of epochs of training
        self.offset = offset  # epoch to start training from
        self.decay_start_epoch = decay_start_epoch  # epoch from which to start lr decay

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
