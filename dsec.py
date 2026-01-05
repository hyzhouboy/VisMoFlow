import os
import cv2
import numpy as np
import torch.utils.data
from utils import disp2pc, project_pc2image, load_flow_png, load_disp_png, load_calib, zero_padding, readFlowDSEC
from augmentation import joint_augmentation
from event_utils.inference_utils import events_to_voxel_grid
from event_utils.event_readers import EventSlicer
import h5py
import hdf5plugin
import json
class DSEC(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        # assert os.path.isdir(cfgs.root_dir)

        self.root_dir = cfgs.root_dir
        if 'training' in cfgs.split:
            self.file_name = os.path.join(cfgs.root_dir, 'training.txt')
        else:
            self.file_name = os.path.join(cfgs.root_dir, 'testing.txt')

        self.split = cfgs.split
        self.cfgs = cfgs
        
        self.num_bins = self.cfgs.events.num_bins
        self.time_slice = self.cfgs.events.time_slice

        with open(self.file_name, 'r') as f:
            self.list = f.readlines()
            del self.list[-1]
            f.close()
        
        self.indices = np.arange(len(self.list))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        child_dir = self.list[index].split(' ')[0]
        frame1_id = self.list[index].split(' ')[1]
        frame2_id = self.list[index].split(' ')[2]
        time_1 = self.list[index].split(' ')[3]
        time_2 = self.list[index].split(' ')[4].split('\n')[0]

        f=569.2873535700672
        cx=336.2678413391113
        cy=222.2889060974121

        image1 = cv2.imread(os.path.join(self.root_dir, child_dir, 'images_rectify', frame1_id+'.png'))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, child_dir, 'images_rectify', frame2_id+'.png'))[..., ::-1]
        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]
        
        flow_2d, flow_2d_mask = readFlowDSEC(os.path.join(self.root_dir, child_dir, 'flow', frame1_id+'.png'))

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, child_dir, 'disp0', frame1_id+'.png'))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, child_dir, 'disp1', frame1_id+'.png'))
        mask = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)

        pc1 = disp2pc(disp1, baseline=0.60, f=f, cx=cx, cy=cy)[mask]
        pc2 = disp2pc(disp2, baseline=0.60, f=f, cx=cx, cy=cy, flow=flow_2d)[mask]
        flow_3d = pc2 - pc1
        flow_3d_mask = np.ones(flow_3d.shape[0], dtype=np.float32)

        # remove out-of-boundary regions of pc2 to create occlusion
        image_h, image_w = disp2.shape[:2]
        xy2 = project_pc2image(pc2, image_h, image_w, f, cx, cy, clip=False)
        boundary_mask = np.logical_and(
            np.logical_and(xy2[..., 0] >= 0, xy2[..., 0] < image_w),
            np.logical_and(xy2[..., 1] >= 0, xy2[..., 1] < image_h)
        )
        pc2 = pc2[boundary_mask]

        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1)
        flow_3d = np.concatenate([flow_3d, flow_3d_mask[..., None].astype(np.float32)], axis=-1)

        
        # read events
        ev_file = h5py.File(os.path.join(self.root_dir, child_dir, 'events', 'events.h5'), 'r')
        eventslice = EventSlicer(ev_file)
        event_timestamps = []

        event_timestamps.append(time_1)
        delta_t = (time_2 - time_1) / self.time_slice
        for i in range(self.time_slice - 2):
            t_start_temp = time_1 + delta_t * (i+1)
            event_timestamps.append(t_start_temp)
        event_timestamps.append(time_2)

        events = []
        events_voxel = []
        for idx in range(len(event_timestamps)):
            t_start = event_timestamps[idx]
            t_end  = event_timestamps[idx] + self.cfgs.events.time_delta

            event_slice = eventslice.get_events(t_start, t_end)
            p = event_slice['p'].astype(np.int8)
            t = event_slice['t'].astype(np.float64)
            x = event_slice['x']
            y = event_slice['y']
            p = 2*p - 1

            events_rectified = np.stack([t, x, y, p], axis=-1)
            events.append(events_rectified)
            # 运动融合阶段单独训练的时候直接输入事件的体素
            if self.cfgs.stage == 4:
                event_tensor = events_to_voxel_grid(events_rectified, num_bins=self.num_bins, width=640, height=480).transpose([1, 2, 0])
                
                events_voxel.append(event_tensor)
        
        # read intensity image from events
        if self.cfgs.stage == 2:
            image_im_1 = cv2.imread(os.path.join(self.root_dir, child_dir, 'intensity_images', frame1_id+'.png'), cv2.IMREAD_GRAYSCALE)
            image_im_2 = cv2.imread(os.path.join(self.root_dir, child_dir, 'intensity_images', frame2_id+'.png'), cv2.IMREAD_GRAYSCALE)

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        padding_h, padding_w = 480, 640
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)
        flow_2d = zero_padding(flow_2d, padding_h, padding_w)
        if self.cfgs.stage == 2:
            image_im_1 = zero_padding(image_im_1, padding_h, padding_w)
            image_im_2 = zero_padding(image_im_2, padding_h, padding_w)
        if self.cfgs.stage == 4:
            for i in range(len(events_voxel)):
                events_voxel[i] = zero_padding(events_voxel[i], padding_h, padding_w)

        # data augmentation:不同阶段的增强不同
        if self.cfgs.stage == 1: # 仅增强图像-点云
            # self.cfgs.augmentation = False
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, _intensity, _events, _voxel = joint_augmentation(
                image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation
            )
        elif self.cfgs.stage == 2:  #HDR部分，增强图像和事件及重建图像、点云
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, _intensity, _events, _voxel = joint_augmentation(
                image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation, [image_im_1, image_im_2], events
            )
            image_im_1 = _intensity[0]
            image_im_2 = _intensity[1]
            events = _events
        elif self.cfgs.stage == 4: #运动融合，仅增强图像、体素和点云
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, _intensity, _events, _voxel = joint_augmentation(
                image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation, intensity_image=None, events=None, event_voxel=events_voxel
            )
            # events = _events
            events_voxel = _voxel



        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        # pcs = np.concatenate([pc1, pc2], axis=1)
        # images = np.concatenate([image1, image2], axis=-1)
        # images_im = np.concatenate([image_im_1, image_im_2], axis=-1)

        data_dict['image_1'] = image1.transpose([2, 0, 1])
        data_dict['image_2'] = image2.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pc_1'] = pc1.transpose()
        data_dict['pc_2'] = pc2.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])
        
        # 事件体素:多个时间片段
        if self.cfgs.stage == 4:
            for i in range(len(events_voxel)):
                events_voxel[i] = events_voxel[i].transpose([2, 0, 1])
            data_dict['event_slice_voxel'] = events_voxel
            
        if self.cfgs.stage == 2:
            # 事件流小片段,此处有bug，尺寸不匹配的问题，后面修复
            data_dict['event_slice'] = events
            # 事件重建亮度帧
            data_dict['intensity_image_1'] = image_im_1.transpose([2, 0, 1])
            data_dict['intensity_image_2'] = image_im_2.transpose([2, 0, 1])
        
            # yuv，仅仅用于第三阶段
            image1_yuv = cv2.cvtColor(image1, cv2.COLOR_RGB2YUV)
            image2_yuv = cv2.cvtColor(image2, cv2.COLOR_RGB2YUV)

            ldr_1_y = image1_yuv[:,:,0] # [0.0, 1.0]
            ldr_1_u = image1_yuv[:,:,1] # [-0.5, 0.5]
            ldr_1_v = image1_yuv[:,:,2] # [-0.5, 0.5]
            ldr_1_y = image1_yuv * 2.0 - 1.0
            data_ldr_1_y = ldr_1_y.astype(np.float32)
            
            data_ldr_1_u = ldr_1_u
            data_ldr_1_v = ldr_1_v

            ldr_2_y = image2_yuv[:,:,0] # [0.0, 1.0]
            ldr_2_u = image2_yuv[:,:,1] # [-0.5, 0.5]
            ldr_2_v = image2_yuv[:,:,2] # [-0.5, 0.5]
            ldr_2_y = image2_yuv * 2.0 - 1.0
            data_ldr_2_y = ldr_2_y.astype(np.float32)
            
            data_ldr_2_u = ldr_2_u
            data_ldr_2_v = ldr_2_v
            
            data_dict['ldr_1_y'] = data_ldr_1_y
            data_dict['ldr_2_y'] = data_ldr_2_y

            data_dict['ldr_1_u'] = data_ldr_1_u
            data_dict['ldr_2_u'] = data_ldr_2_u

            data_dict['ldr_1_v'] = data_ldr_1_v
            data_dict['ldr_2_v'] = data_ldr_2_v


        return data_dict

