import torch
import torch.nn as nn
from .base import FlowModel
from .vismoflow_core import VisMoFlow_Core
from .losses import calc_sequence_loss_2d, calc_sequence_loss_3d, calc_kl_divergence, calc_slice_loss_census
from .utils import InputPadder
from .ids import paral2persp, persp2paral


class VisMoFlow(FlowModel):
    def __init__(self, cfgs):
        super(VisMoFlow, self).__init__()
        self.cfgs = cfgs
        self.core = VisMoFlow_Core(cfgs)

    def train(self, mode=True):
        self.training = mode

        for module in self.children():
            module.train(mode)

        if self.cfgs.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        return self

    def eval(self):
        return self.train(False)

    def forward(self, inputs):
        images = inputs['images'].float()
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']

        # events voxel 1-2
        event_voxels = inputs['voxels'].float()
        
        # pad input shape to a multiple of 8
        padder = InputPadder(images.shape, x=8)
        image1, image2 = padder.pad(images[:, :3], images[:, 3:])

        norm_mean = torch.tensor([123.675, 116.280, 103.530], device=images.device)
        norm_std = torch.tensor([58.395, 57.120, 57.375], device=images.device)
        image1 = image1 - norm_mean.reshape(1, 3, 1, 1)
        image2 = image2 - norm_mean.reshape(1, 3, 1, 1)
        image1 = image1 / norm_std.reshape(1, 3, 1, 1)
        image2 = image2 / norm_std.reshape(1, 3, 1, 1)


        event1_voxel, event2_voxel = padder.pad(event_voxels[:, :3], event_voxels[:, 3:])

        slice_paired_voxels = []
        if self.cfgs.cv_densefication:
            for i in range(len(inputs['slices_voxels']) - 1):
                voxel_1_, voxel_2_ = padder.pad(inputs['slices_voxels'][i].float(), inputs['slices_voxels'][i+1].float())
                slice_paired_voxels.append([voxel_1_, voxel_2_])

        # norm_mean = torch.tensor([123.675, 116.280, 103.530], device=event1_voxel.device)
        # norm_std = torch.tensor([58.395, 57.120, 57.375], device=event1_voxel.device)
        # event1_voxel = event1_voxel - norm_mean.reshape(1, 3, 1, 1)
        # event2_voxel = event2_voxel - norm_mean.reshape(1, 3, 1, 1)
        # event1_voxel = event1_voxel / norm_std.reshape(1, 3, 1, 1)
        # event2_voxel = event2_voxel / norm_std.reshape(1, 3, 1, 1)

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': image1.shape[-2],  # 544
            'sensor_w': image1.shape[-1],  # 960
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }
        paral_cam_info = {
            'projection_mode': 'parallel',
            'sensor_h': round(image1.shape[-2] / 32),
            'sensor_w': round(image1.shape[-1] / 32),
            'cx': (round(image1.shape[-1] / 32) - 1) / 2,
            'cy': (round(image1.shape[-2] / 32) - 1) / 2,
        }
        pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
        pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)

        input_data = dict()
        input_data['image1'] = image1
        input_data['image2'] = image2
        input_data['pc1'] = pc1
        input_data['pc2'] = pc2
        input_data['event1_voxel'] = event1_voxel
        input_data['event2_voxel'] = event2_voxel
        input_data['paral_cam_info'] = paral_cam_info
        if self.cfgs.cv_densefication:
            input_data['paired_event_slices'] = slice_paired_voxels

        if self.cfgs.cv_densefication:
            flow_2d_preds, flow_3d_preds, flow_ev2d_preds, slice_flow_ev2d_preds = self.core(input_data)
        else:
            flow_2d_preds, flow_3d_preds, flow_ev2d_preds = self.core(input_data)

        for i in range(len(flow_2d_preds)):
            flow_2d_preds[i] = padder.unpad(flow_2d_preds[i])

        for i in range(len(flow_3d_preds)):
            flow_3d_preds[i] = paral2persp(pc1 + flow_3d_preds[i], persp_cam_info, paral_cam_info) -\
                               paral2persp(pc1, persp_cam_info, paral_cam_info)

        for i in range(len(flow_ev2d_preds)):
            flow_ev2d_preds[i] = padder.unpad(flow_ev2d_preds[i])


        final_flow_2d = flow_2d_preds[-1]
        final_flow_3d = flow_3d_preds[-1]
        final_flow_ev2d = flow_ev2d_preds[-1]

        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d, 'flow_ev2d': final_flow_ev2d}

        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()

        # kl散度
        loss_kl = 0
        if self.cfgs.kl_alignment:
            motion_feats_rgb, motion_feats_event, motion_feats_lidar = self.core.get_motion_feats()
            for i in range(len(motion_feats_rgb)):
                kl_div_level = calc_kl_divergence(motion_feats_rgb[i], motion_feats_event[i], motion_feats_lidar[i])
                loss_kl += kl_div_level
            loss_kl = loss_kl / len(motion_feats_rgb)

        # calculate losses
        loss_2d = calc_sequence_loss_2d(flow_2d_preds, target_2d, cfgs=self.cfgs.loss2d)
        loss_3d = calc_sequence_loss_3d(flow_3d_preds, target_3d, cfgs=self.cfgs.loss3d)
        loss_ev2d = calc_sequence_loss_2d(flow_ev2d_preds, target_2d, cfgs=self.cfgs.loss2d)

        # temporal dense loss: unsupervised photometric loss
        loss_temporal = 0
        if self.cfgs.cv_densefication:
            for i in range(len(slice_paired_voxels)):
                loss_2d_pho = calc_slice_loss_census(slice_paired_voxels[i][0], slice_paired_voxels[i][1], slice_flow_ev2d_preds[i])
                loss_temporal += loss_2d_pho
            loss_temporal /= len(slice_paired_voxels)

        self.loss = loss_2d + loss_3d + loss_ev2d + 0.1*loss_kl + 0.3*loss_temporal

        self.update_metrics('loss', self.loss)
        self.update_metrics('loss2d', loss_2d)
        self.update_metrics('loss3d', loss_3d)
        self.update_metrics('loss_ev2d', loss_ev2d)
        if self.cfgs.kl_alignment:
            self.update_metrics('loss_kl', loss_kl)
        if self.cfgs.cv_densefication:
            self.update_metrics('loss_temporal', loss_temporal)
        self.update_2d_metrics(final_flow_2d, target_2d)
        self.update_ev2d_metrics(final_flow_ev2d, target_2d)
        self.update_3d_metrics(final_flow_3d, target_3d)

        if 'occ_mask_3d' in inputs:
            self.update_3d_metrics(final_flow_3d, target_3d, inputs['occ_mask_3d'])

        return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d, 'flow_ev2d': final_flow_ev2d}

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['epe2d'] < best_metrics['epe2d']
