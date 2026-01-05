from __future__ import print_function, division
import sys
# sys.path.append('core')
import math
from typing import Dict, Tuple
from pathlib import Path
import weakref
from skimage.transform import rotate, warp
import h5py
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch 
import open3d
from skimage.color import hsv2rgb
from PIL import Image

def flow2rgb(flow_map, max_value):
    # 2 x H x w
    h, w, _ = flow_map.shape
    flow_u, flow_v = flow_map[:, :, 0], flow_map[:, :, 1]
    n = 8
    mag = np.sqrt(np.sum(np.square(flow_map), axis=2))
    angle = np.arctan2(flow_v, flow_u)

    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_value, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)
    im_hsv = np.stack((im_h, im_s, im_v), axis=2)

    im = hsv2rgb(im_hsv)
    return im


def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc

def depth2pc(depth, f, cx, cy, flow=None):
    h, w = depth.shape
    # depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc

if __name__ == '__main__':

    # width = 1280
    # height = 720

    cam_K = np.array([[2.383032216559313e+03, 0.0000000000000000000, 1.042903625901269e+03, 0.0000000000000000000],
                  [0.0000000000000000000, 2.384081147246841e+03, 7.919197184630658e+02, 0.0000000000000000000],
                  [0.0000000000000000000, 0.0000000000000000000, 1.0000000000000000000, 0.0000000000000000000]])
    
    # 重定义平移向量，微调平移向量参数即可--x y z
    translation_vector = np.array([ 0.250000, -0.263000, 0.184233])
    f = 2383.0
    cx = 1042.904
    cy = 791.920
    depth_max = 120.0

    # 读取原始点云
    pcd = open3d.io.read_point_cloud('C:/Users/Wade/Desktop/SceneFlow_CVPR/SceneFlow_1016/demo_img/velo_pcd/00001_0.pcd')
    lidar_pcd = np.asarray(pcd.points)
    print(lidar_pcd.shape)
    
    # 自动驾驶场景下：lidar坐标系点云->相机坐标系点云
    lidar2camera_pcd = lidar_pcd + translation_vector
    lidar2camera_pcd = lidar2camera_pcd.transpose()
    camera_pcd = np.zeros_like(lidar2camera_pcd)
    camera_pcd[0, :] = lidar2camera_pcd[0, :]
    camera_pcd[1, :] = -1. *lidar2camera_pcd[2, :]
    camera_pcd[2, :] = lidar2camera_pcd[1, :]
    # camera_points = np.concatenate([camera_pointcloud[0, :], camera_pointcloud[1, :], camera_pointcloud[1, :]], 1)       
    camera_pcd = camera_pcd.transpose()   

    # 由于camera仅存在前视角度，后视角度点云过滤
    mask = (camera_pcd[:, 2] > 0)
    camera_pcd = camera_pcd[mask]
 
    # 读取图片以获取相机分辨率
    image = cv2.imread('C:/Users/Wade/Desktop/SceneFlow_CVPR/SceneFlow_1016/demo_img/image/00001_0.bmp')  #路径自己替换
    h, w = image.shape[0], image.shape[1]

    # 相机坐标系变换到像素坐标系
    # depth = baseline * f / (disp + 1e-5)
    pixel_coordinate = np.zeros_like(camera_pcd)
    pixel_coordinate[:, 0] = camera_pcd[:, 0] * f / camera_pcd[:, 2] + cx 
    pixel_coordinate[:, 1] = camera_pcd[:, 1] * f / camera_pcd[:, 2] + cy
    pixel_coordinate[:, 2] = camera_pcd[:, 2]

    # 过滤图像坐标系下边缘以外多余的点云数据
    mask = (pixel_coordinate[:, 0] > 0) & (pixel_coordinate[:, 0] < w) & (pixel_coordinate[:, 1] > 0) & (pixel_coordinate[:, 1] < h)
    camera_pcd = camera_pcd[mask]
    pixel_coordinate = pixel_coordinate[mask]
    print(camera_pcd.shape)
    
    # 显示点云
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(camera_pcd)
    # open3d.io.write_point_cloud(os.path.join(os.path.join(path, 'enhanced_point_cloud'), files[index][:-4]+'.pcd'), point_cloud)
    open3d.visualization.draw_geometries([point_cloud])
    
    # 深度图获取
    depth = np.zeros((h, w))
    for point in pixel_coordinate:
        depth[int(point[1]), int(point[0])] = point[2]
    
    # 过滤超过120m单位的深度
    mask = (depth[:, :] < depth_max)
    depth = depth * mask
    # 存储深度图(注意：换算成了cm单位进行存储)
    cv2.imwrite('C:/Users/Wade/Desktop/SceneFlow_CVPR/SceneFlow_1016/demo_img/00001.png', (100*depth).astype(np.uint16)) #路径自己替换

    # 存储深度的伪彩色图
    gray_depth_image = (255 * (depth.astype(np.uint16) - np.min(depth)) / (np.max(depth) - np.min(depth))).astype(np.uint8)
    im_color=cv2.applyColorMap(gray_depth_image,cv2.COLORMAP_BONE)
    im_color=Image.fromarray(im_color)
    im_color.save('C:/Users/Wade/Desktop/SceneFlow_CVPR/SceneFlow_1016/demo_img/00001_color.png') #路径自己替换
    