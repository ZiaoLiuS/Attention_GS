#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from math import exp
import numpy as np

import cv2
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian_kernel(kernel_size, sigma):
    """
    生成一个二维高斯核。

    参数:
        kernel_size (int): 高斯核的大小（必须是奇数）。
        sigma (float): 高斯分布的标准差。

    返回:
        torch.Tensor: 二维高斯核，形状为 [kernel_size, kernel_size]。
    """
    # 生成一维高斯核
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()  # 归一化

    # 生成二维高斯核
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]  # 外积
    return gauss_2d

def gaussian_blur(input_tensor, kernel_size, sigma):
    """
    对输入的 3 通道张量进行高斯模糊。

    参数:
        input_tensor (torch.Tensor): 输入张量，形状为 [3, m, n]。
        kernel_size (int): 高斯核的大小（必须是奇数）。
        sigma (float): 高斯分布的标准差。

    返回:
        torch.Tensor: 高斯模糊后的张量，形状为 [3, m, n]。
    """
    # 检查输入张量的形状
    if input_tensor.dim() != 3 or input_tensor.size(0) != 3:
        raise ValueError("输入张量的形状必须是 [3, m, n]")

    # 生成高斯核
    kernel = gaussian_kernel(kernel_size, sigma)

    # 将高斯核扩展到 4D 张量 [1, 1, kernel_size, kernel_size]
    kernel = kernel[None, None, :, :].to(input_tensor.device)

    # 对每个通道进行卷积
    blurred_tensor = []
    for channel in input_tensor:
        # 扩展为 4D 张量 [1, 1, m, n]
        channel = channel[None, None, :, :]
        # 使用卷积操作进行高斯模糊
        blurred_channel = F.conv2d(channel, kernel, padding=kernel_size // 2)
        blurred_tensor.append(blurred_channel.squeeze())

    # 合并通道
    return torch.stack(blurred_tensor)

def adjustable_decay(iteration, n, steepness=10, midpoint_ratio=0.5):
    x = 2 * steepness * (iteration / n - midpoint_ratio)
    return 1 / (1 + np.exp(x))

def attention_loss(network_output, gt, iteration):
    weights = torch.abs((network_output - gt))
    layer_norm = nn.LayerNorm(weights.size()[1:]).cuda() 

    weights = layer_norm(weights-weights.min())*(1-adjustable_decay(iteration,30000,midpoint_ratio=0.25)) + 1.0
    
    
    weights = weights

    # 转换为灰度图像
    gray_tensor = 0.2989 * weights[0] + 0.5870 * weights[1] + 0.1140 * weights[2]
    weights = torch.stack([gray_tensor, gray_tensor, gray_tensor], dim=0)


    return torch.abs(weights * (network_output - gt)).mean() ,weights


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def rgb_to_grayscale(tensor):
    grayscale = 0.2989 * tensor[0] + 0.5870 * tensor[1] + 0.1140 * tensor[2]
    return grayscale.unsqueeze(0) 

def sobel_gradient(tensor):
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_x = sobel_x.to(tensor.device)
    sobel_y = sobel_y.to(tensor.device)
    
    grad_x = F.conv2d(tensor.unsqueeze(0), sobel_x, padding=1).squeeze(0)
    grad_y = F.conv2d(tensor.unsqueeze(0), sobel_y, padding=1).squeeze(0)
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    return gradient_magnitude

def canny_edge(tensor):
    image_np = tensor.detach().squeeze(0).cpu().numpy()
    edges = cv2.Canny((image_np * 255).astype(np.uint8), 50, 150)  
    return torch.from_numpy(edges).unsqueeze(0).to(tensor.device) / 255.0

def cross_entropy_weight_loss(network_output, gt,iteration):
    network_output_gray = rgb_to_grayscale(network_output)  # 形状变为 (1, m, n)
    gt_gray = rgb_to_grayscale(gt)  # 形状变为 (1, m, n)
    
    # Sobel (batch_size, 1, H, W)
    network_output_grad = sobel_gradient(network_output_gray)
    gt_grad = sobel_gradient(gt_gray)
    
    # # canny (batch_size, 1, H, W)
    # network_output_grad = canny_edge(network_output_gray)
    # gt_grad = canny_edge(gt_gray)
    
    grad_diff = torch.abs(network_output_grad - gt_grad)
    weights = (grad_diff - grad_diff.min()) / (grad_diff.max() - grad_diff.min() + 1e-8) 
    weights = weights.expand(3, -1, -1).detach()
    
    l1_loss = torch.abs(network_output - gt)
    weighted_loss = (weights *(adjustable_decay(iteration,30000,midpoint_ratio=0.25))).detach() * l1_loss
    
    return torch.mean(weighted_loss)
