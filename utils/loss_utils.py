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
from torch.autograd import Variable
from math import exp
import cv2

def space_loss(tensor_image1,tensor_image2):
    
    #tensor_image1 = 255-tensor_image1
    tensor_image1 = tensor_image1.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    tensor_image2 = tensor_image2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    
    #tensor_image1 = 255-tensor_image1
    tensor_image2 = 255-tensor_image2
    # 对灰度图像张量进行阈值化处理
    
    threshold = torch.tensor(240).cuda()  # 阈值，替换为适当的阈值
    # 将阈值移动到与图像张量相同的设备上
    threshold = threshold.to(tensor_image1.device)

    # 创建用于替代的张量，并将其移动到与图像张量相同的设备上
    ones_tensor = torch.tensor(255).to(tensor_image1.device)
    zeros_tensor = torch.tensor(0).to(tensor_image1.device)
    binary_tensor = torch.where(tensor_image1 < threshold, ones_tensor, zeros_tensor)
    #binary_tensor = 255-binary_tensor
    binary_tensor = binary_tensor/255
    masked_image = tensor_image2 * binary_tensor
    #total_sum = torch.sum(masked_image)/(1024*1024)
    '''
    tensor_image2 = 255-tensor_image2
    #total_sum = torch.sum(masked_image)
    #tensor_image1 = tensor_image1.numpy()
    a = l1_loss(tensor_image2,masked_image)
    tensor_image2 = tensor_image2.cpu().detach().numpy()
    masked_image = masked_image.cpu().detach().numpy()
    '''
    masked_image = 255-masked_image
    tensor_image2 = 255-tensor_image2
    '''
    binary_tensor = binary_tensor*255
    binary_tensor = binary_tensor.cpu().detach().numpy()
    tensor_image1 = tensor_image1.cpu().detach().numpy()
    tensor_image2 = tensor_image2.cpu().detach().numpy()
    masked_image = masked_image.cpu().detach().numpy()
    cv2.imwrite("/data5/zhangshuai/gaussian-splatting6/newloss/image1test.jpg", tensor_image1) 
    cv2.imwrite("/data5/zhangshuai/gaussian-splatting6/newloss/image2test.jpg", tensor_image2) 
    cv2.imwrite("/data5/zhangshuai/gaussian-splatting6/newloss/binary_tensortest.jpg", binary_tensor) 
    cv2.imwrite("/data5/zhangshuai/gaussian-splatting6/newloss/masked_imagetest.jpg", masked_image)
    ''' 
    return l1_loss(tensor_image2,masked_image)

def space_loss_yuan(tensor_image1,tensor_image2):
    
    #tensor_image1 = 255-tensor_image1
    tensor_image1 = tensor_image1.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    tensor_image2 = tensor_image2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    tensor_image2 = 255-tensor_image2
    # 对灰度图像张量进行阈值化处理
    threshold = torch.tensor(254).cuda()  # 阈值，替换为适当的阈值
    # 将阈值移动到与图像张量相同的设备上
    threshold = threshold.to(tensor_image1.device)

    # 创建用于替代的张量，并将其移动到与图像张量相同的设备上
    ones_tensor = torch.tensor(255).to(tensor_image1.device)
    zeros_tensor = torch.tensor(0).to(tensor_image1.device)
    binary_tensor = torch.where(tensor_image1 > threshold, ones_tensor, zeros_tensor)
    binary_tensor = binary_tensor/255
    masked_image = tensor_image2 * binary_tensor
    total_sum = torch.sum(masked_image)/(1024*1024)
    #total_sum = torch.sum(masked_image)
    return total_sum

def scale_loss(val):
    #value = 0
    #for i in range(1,len(val[0])):
    #    value += torch.abs(val[:,i]-val[:,i-1])
    max_values, _ = torch.max(val, dim=1)
    min_values, _ = torch.min(val, dim=1)
    result = max_values - min_values

    return result.float().mean()


def table_loss(val):
    value = 0
    for i in range(1,len(val[0])):
        value += torch.abs(val[:,i]-val[:,i-1])
    return value.float().mean()

def table_loss4(val):
    value = 0
    for i in range(len(val[0])):
        value += val[:,i]
    return value.float().mean()

def table_loss3(val,view_time):
    view_time = int(view_time*5)
    value = 0
    value = torch.abs(val[:,view_time]-val[:,view_time-1])
    return value.float().mean()

def table_loss2(val):
    value = torch.abs(val[:,1]-val[:,0])
    for i in range(1,len(val[0])):
        #print(torch.abs(val[:,i]-val[:,i-1]).shape)
        #value =max(value,torch.abs(val[:,i]-val[:,i-1]))
        #print(value)
        a = torch.stack([value,torch.abs(val[:,i]-val[:,i-1])])
        value,_ = torch.max(a, dim=1)
    return value.float().mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

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


if __name__=='__main__':
    val = torch.tensor([[1,1,1,1,1],[1,5,0,0,0]])
    view_time = 0.5
    #print(table_loss3(val,view_time))
    print(table_loss4(val))

