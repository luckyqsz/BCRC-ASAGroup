import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(config):  # config is a dict
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=False, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=False, transform=transform)
    '''
    root : processed/training.pt 和 processed/test.pt 的主目录
    train : True = 训练集, False = 测试集
    download : True = 从互联网上下载数据集，并把数据集放在 root 目录下. 如果数据集
    之前下载过，将处理过的数据（minist.py中有相关函数）放在 processed 文件夹下。
    '''

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader
    # 两者具有相同的batchsize
