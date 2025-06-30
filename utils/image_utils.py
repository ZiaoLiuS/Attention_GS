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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

import lpips

# Initialize the LPIPS model
lpips_model = lpips.LPIPS(net='alex').to('cuda')

def calculate_lpips(img1, img2, device='cuda'):
    """
    Calculate LPIPS between two images.
    
    Args:
        img1 (torch.Tensor): Input image 1 (range [0, 1], shape [B, C, H, W]).
        img2 (torch.Tensor): Input image 2 (range [0, 1], shape [B, C, H, W]).
        net (str): Network to use for LPIPS ('alex', 'vgg', or 'squeeze').
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        lpips_value (torch.Tensor): LPIPS score.
    """
    # Ensure images are in the range [0, 1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        raise ValueError("Input images should be in the range [0, 1].")
    
    # Move images to the specified device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # # Initialize the LPIPS model
    # lpips_model = lpips.LPIPS(net=net).to(device)
    
    # Calculate LPIPS
    lpips_value = lpips_model(img1, img2)
    
    return lpips_value

from torchmetrics import StructuralSimilarityIndexMeasure

def calculate_ssim(img1, img2, device='cuda'):
    """
    Calculate SSIM between two images.
    
    Args:
        img1 (torch.Tensor): Input image 1 (range [0, 1], shape [B, C, H, W]).
        img2 (torch.Tensor): Input image 2 (range [0, 1], shape [B, C, H, W]).
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        ssim_value (torch.Tensor): SSIM score.
    """
    # Ensure images are in the range [0, 1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        raise ValueError("Input images should be in the range [0, 1].")
    
    # Move images to the specified device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Initialize SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Calculate SSIM
    ssim_value = ssim_metric(img1, img2)
    
    return ssim_value