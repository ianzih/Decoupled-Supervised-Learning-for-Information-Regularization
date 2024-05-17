import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
        
        
# for convolutional neural networks
def conv_layer_bn(in_channels: int, out_channels: int, activation: nn.Module, stride: int=1, bias: bool=False, padding: int=1, kernel_size: int=3) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, bias = bias, padding = padding)
    bn = nn.BatchNorm2d(out_channels)
    if activation == None:
        return nn.Sequential(conv, bn)
    return nn.Sequential(conv, bn, activation)

def conv_1x1_bn(in_channels: int, out_channels: int, activation: nn.Module = None, stride: int=1, bias: bool=False) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = bias)
    bn = nn.BatchNorm2d(out_channels)
    if activation == None:
        return nn.Sequential(conv, bn)
    return nn.Sequential(conv, bn, activation)


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
    
def add_noise_cifar(loader, class_num, noise_rate):
    """ 參考自 https://github.com/PaulAlbert31/LabelNoiseCorrection """
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_rate)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(class_num)))
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        if transform1 != None and transform2 == None:
            self.transform2 = transform1
        else:
            self.transform2 = transform2

    def __call__(self, x):
        if self.transform1 == None:
            return [x, x]
        return [self.transform1(x), self.transform2(x)]
