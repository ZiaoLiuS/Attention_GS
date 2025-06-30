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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def split_frequency_components(image, low_freq_ratio=0.3, mid_freq_ratio=1.0):
    """
    输入: 
        image: 形状为 (3, m, n) 的 Tensor，表示 RGB 图像。
        low_freq_ratio: 低频部分的比例。
        mid_freq_ratio: 中频部分的比例。
    输出:
        low_freq: 低频部分的灰度图，形状为 (3, m, n)。
        mid_freq: 中频部分的灰度图，形状为 (3, m, n)。
        high_freq: 高频部分的灰度图，形状为 (3, m, n)。
    """
    assert image.dim() == 3 and image.shape[0] == 3, "输入必须是形状为 (3, m, n) 的 Tensor"
    
    # 获取图像的高度和宽度
    m, n = image.shape[1], image.shape[2]
    
    # 初始化输出
    low_freq = torch.zeros_like(image)
    mid_freq = torch.zeros_like(image)
    high_freq = torch.zeros_like(image)
    
    # 对每个通道分别处理
    for c in range(3):
        channel = image[c]  # 获取当前通道
        
        # 傅里叶变换
        fft_channel = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft_channel)
        
        # 中心坐标
        center_m, center_n = m // 2, n // 2
        
        # 低频掩码
        low_freq_size_m = int(low_freq_ratio * m)
        low_freq_size_n = int(low_freq_ratio * n)
        mask_low = torch.zeros_like(fft_shifted)
        mask_low[center_m - low_freq_size_m // 2:center_m + low_freq_size_m // 2,
                 center_n - low_freq_size_n // 2:center_n + low_freq_size_n // 2] = 1
        
        # 中频掩码
        mid_freq_size_m = int(mid_freq_ratio * m)
        mid_freq_size_n = int(mid_freq_ratio * n)
        mask_mid = torch.zeros_like(fft_shifted)
        mask_mid[center_m - mid_freq_size_m // 2:center_m + mid_freq_size_m // 2,
                 center_n - mid_freq_size_n // 2:center_n + mid_freq_size_n // 2] = 1
        mask_mid -= mask_low  # 排除低频部分
        
        # 高频掩码
        mask_high = 1 - mask_low - mask_mid
        
        # 应用掩码
        fft_low = fft_shifted * mask_low
        fft_mid = fft_shifted * mask_mid
        fft_high = fft_shifted * mask_high
        
        # 逆傅里叶变换
        low_freq[c] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_low)))
        mid_freq[c] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_mid)))
        high_freq[c] = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_high)))
    
    return low_freq, mid_freq, high_freq