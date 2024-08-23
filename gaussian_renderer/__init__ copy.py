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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, view_time= None, iteration):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    time_opacity = pc.get_time_opacity

    #print('opacity:',opacity[400])
    #print('400:',time_opacity[400])
    #print('opacity:',opacity[1400])
    #print('1400:',time_opacity[1400])
    #print(opacity[400])

    
    view_time2 = view_time*4
    x2 = math.ceil(view_time2)
    x1 = int(view_time2)
    y1 = time_opacity[:,x1]
    y2 = time_opacity[:,x2]


    '''
    # 切片获取第二维前 y 项
    sub_tensor = time_opacity[:, :x1]
    # 沿第二维度求和，保持维度为 (1500, 1)
    sum_tensor = torch.sum(sub_tensor, dim=1)
    print(sum_tensor.shape)
    panyi = ((view_time-x1)*y2-(view_time-x1)*y1)/((x2-x1)+0.00000001)+sum_tensor
    '''
    
    if(x1>=1 and x2<=3):
        x3 = x1-1
        x4 = x2+1
        y3 = time_opacity[:,x3]
        y4 = time_opacity[:,x4]
        #panyi = 0.5*(((view_time2-x1)*y2-(view_time2-x1)*y1)/((x2-x1)+0.00000001))+0.5*y1+0.5*y3+0.5*(((view_time-x3)*y4-(view_time-x3)*y3)/((x4-x3)+0.00000001))
        #panyi = 0.5*(((view_time2-x3)*y4-(view_time2-x3)*y3)/((x4-x3)+0.00000001)+y3+((view_time2-x1)*y2-(view_time2-x1)*y1)/((x2-x1)+0.00000001)+y1)
        #panyi = y1+((view_time2-x1)*y2-(view_time2-x1)*y1)/((x2-x1)+0.00000001)
        panyi = y1+((view_time2-x1)*y2-(view_time2-x1)*y1)/((x2-x1)+0.00000001)
    else:
        panyi = y1+((view_time2-x1)*y2-(view_time2-x1)*y1)/((x2-x1)+0.00000001)
    
    #print((view_time-x3)*y4-(view_time-x3)*y3)
    panyi = panyi.unsqueeze(1)

    #print(view_time)
    #print(time_opacity[0])
    #print(opacity[0],panyi[0])
    #print(x1,x2,x3,x4)
    #print(time_opacity[0])
    opacity = opacity+panyi
    #opacity = opacity
    


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    #print('scales_shape:',scales[0])            #1000*3
    #print('rotations_shape:',rotations[0])      #1000*4

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    
    #取消彩色 改为灰度
    new_shs = shs[:, :, 0:1]
    new_shs = new_shs.expand(new_shs.shape[0], 16, 3)
    #print(means3D.shape)
    #print(shs.shape)
    #print(opacity.shape)
    #print(a)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = new_shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
