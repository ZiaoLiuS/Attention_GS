import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

def low_pass_filter(input_tensor):
    # 定义一个3x3的平均滤波器
    kernel = torch.ones(1, 1, 3, 3) / 9.0
    kernel = kernel.repeat(3, 1, 1, 1)  # 重复核以匹配输入通道数
    
    # 使用卷积操作进行滤波
    output_tensor = F.conv2d(input_tensor, kernel.cuda(), padding=1, groups=3)
    
    return output_tensor

def high_pass_filter(input_tensor):
    # 定义一个3x3的拉普拉斯滤波器
    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32).view(1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)  # 重复核以匹配输入通道数
    
    # 使用卷积操作进行滤波
    output_tensor = F.conv2d(input_tensor, kernel.cuda(), padding=1, groups=3)
    
    return output_tensor

