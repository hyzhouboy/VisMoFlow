import torch
import torch.nn as nn
from .clfm import CLFM
from .clfm_rel import CLFM_REL, FusionAwareInterp
from .raft_core import RAFTCore
from .camliraft_l_core import CamLiRAFT_L_Core
from .utils import mesh_grid, k_nearest_neighbor, knn_interpolation, backwarp_3d, project_pc2image, build_pc_pyramid


class VisMoFlow_Core(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.corr_levels = 4
        self.corr_radius = 4

        # Name is important!
        self.branch_2d = RAFTCore(cfgs)
        self.branch_3d = CamLiRAFT_L_Core(cfgs)

        # event brach
        self.branch_2d_event = RAFTCore(cfgs)

        # fuse
        if self.cfgs.event_fusion:
            if self.cfgs.fuse_fnet:
                self.clfm_fnet = CLFM_REL(128, 128, norm='batch_norm')
            if self.cfgs.fuse_cnet:
                self.clfm_cnet = CLFM_REL(128, 128, norm='batch_norm')
            if self.cfgs.fuse_corr:
                self.clfm_corr = CLFM_REL(81 * 4, 128)
            if self.cfgs.fuse_motion:
                self.clfm_motion = CLFM_REL(128, 128)
            if self.cfgs.fuse_hidden:
                self.clfm_hidden = CLFM_REL(128, 128)
        else:      
            if self.cfgs.fuse_fnet:
                self.clfm_fnet = CLFM(128, 128, norm='batch_norm')
            if self.cfgs.fuse_cnet:
                self.clfm_cnet = CLFM(128, 128, norm='batch_norm')
            if self.cfgs.fuse_corr:
                self.clfm_corr = CLFM(81 * 4, 128)
            if self.cfgs.fuse_motion:
                self.clfm_motion = CLFM(128, 128)
            if self.cfgs.fuse_hidden:
                self.clfm_hidden = CLFM(128, 128)
        
        self.motion_feats_rgb = []
        self.motion_feats_lidar = []
        self.motion_feats_event = []
            

    def forward(self, input_data):
        image1, image2, pc1, pc2, event_voxel1, event_voxel2, camera_info = \
            input_data['image1'], input_data['image2'], input_data['pc1'], input_data['pc2'], \
            input_data['event1_voxel'], input_data['event2_voxel'], input_data['paral_cam_info']
            
        # build point pyramid using FPS
        xyzs1, xyzs2, _, _ = build_pc_pyramid(
            pc1, pc2, [4096, 2048, 1024, 512, 256]
        )

        # feature and context network (2D)
        feat1_2d = self.branch_2d.fnet(image1)
        feat2_2d = self.branch_2d.fnet(image2)
        featc_2d = self.branch_2d.cnet(image1)

        # feature and context network (3D)
        feat1_3d = self.branch_3d.fnet(xyzs1[:3])[2]
        feat2_3d = self.branch_3d.fnet(xyzs2[:3])[2]
        featc_3d = self.branch_3d.cnet(xyzs1[:3])[2]

        # feature and context network (2D event)
        evfeat1_2d = self.branch_2d_event.fnet(event_voxel1)
        evfeat2_2d = self.branch_2d_event.fnet(event_voxel2)
        evfeatc_2d = self.branch_2d_event.cnet(event_voxel1)

        # [2048, 1024, 512, 256]
        xyzs1, xyzs2 = xyzs1[2:], xyzs2[2:]
        xyz1, xyz2 = xyzs1[0], xyzs2[0]  # 2048

        # project point cloud to image
        sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
        uv1 = project_pc2image(xyz1, camera_info)
        uv2 = project_pc2image(xyz2, camera_info)
        uv1[:, 0] *= (feat1_2d.shape[-1] - 1) / (sensor_w - 1)
        uv1[:, 1] *= (feat1_2d.shape[-2] - 1) / (sensor_h - 1)
        uv2[:, 0] *= (feat2_2d.shape[-1] - 1) / (sensor_w - 1)
        uv2[:, 1] *= (feat2_2d.shape[-2] - 1) / (sensor_h - 1)

        # fuse feature network
        if self.cfgs.fuse_fnet:
            if self.cfgs.event_fusion:
                feat1_2d, feat1_3d = self.clfm_fnet(uv1, feat1_2d, feat1_3d, evfeat1_2d)
                feat2_2d, feat2_3d = self.clfm_fnet(uv2, feat2_2d, feat2_3d, evfeat2_2d)
            else:
                feat1_2d, feat1_3d = self.clfm_fnet(uv1, feat1_2d, feat1_3d)
                feat2_2d, feat2_3d = self.clfm_fnet(uv2, feat2_2d, feat2_3d)

        # fuse context network
        if self.cfgs.fuse_cnet:
            if self.cfgs.event_fusion:
                featc_2d, featc_3d = self.clfm_cnet(uv1, featc_2d, featc_3d, evfeatc_2d)
            else:
                featc_2d, featc_3d = self.clfm_cnet(uv1, featc_2d, featc_3d)

        # init hidden state (2D)
        featc_2d = self.branch_2d.cnet_aligner(featc_2d)
        h_2d, x_2d = torch.split(featc_2d, [128, 128], dim=1)
        h_2d = torch.tanh(h_2d)
        x_2d = torch.relu(x_2d)

        # init hidden state (3D)
        featc_3d = self.branch_3d.cnet_aligner(featc_3d)
        h_3d, x_3d = torch.split(featc_3d, [128, 128], dim=1)
        h_3d = torch.tanh(h_3d)
        x_3d = torch.relu(x_3d)

        # init hidden state (2D event)
        evfeatc_2d = self.branch_2d_event.cnet_aligner(evfeatc_2d)
        h_ev2d, x_ev2d = torch.split(evfeatc_2d, [128, 128], dim=1)
        h_ev2d = torch.tanh(h_ev2d)
        x_ev2d = torch.relu(x_ev2d)

        # build all-pair correlation
        self.branch_2d.correlation.build_cost_volume_pyramid(feat1_2d, feat2_2d)
        self.branch_3d.correlation.build_cost_volume_pyramid(feat1_3d, feat2_3d, xyzs2)

        # build all-pair correlation of event
        self.branch_2d_event.correlation.build_cost_volume_pyramid(evfeat1_2d, evfeat2_2d)


        # precompute k-nn indices
        knn_indices = k_nearest_neighbor(xyz1, xyz1, k=32)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        # init
        bs, _, image_h, image_w = image1.shape
        grid_coords = mesh_grid(bs, image_h//8, image_w//8, device=image1.device)
        flow_2d_pred = torch.zeros_like(grid_coords)
        flow_3d_pred = torch.zeros_like(xyz1)
        xyzs2_warp = xyzs2

        ev_grid_coords = mesh_grid(bs, image_h//8, image_w//8, device=event_voxel1.device)
        flow_ev2d_pred = torch.zeros_like(ev_grid_coords)

        flow_2d_preds, flow_3d_preds, flow_ev2d_preds = [], [], []
        self.motion_feats_rgb.clear()
        self.motion_feats_lidar.clear()
        self.motion_feats_event.clear()
        for it in range(n_iters):
            if it > 0:
                flow_2d_pred = flow_2d_pred.detach()
                flow_3d_pred = flow_3d_pred.detach()
                xyzs2_warp = [backwarp_3d(xyz1, xyz2_lvl, flow_3d_pred) for xyz2_lvl in xyzs2]
                flow_ev2d_pred = flow_ev2d_pred.detach()

            # correlation
            corr2d = self.branch_2d.correlation(grid_coords + flow_2d_pred)
            corr3d = self.branch_3d.correlation(xyz1, xyzs2_warp)
            corr2d_ev = self.branch_2d_event.correlation(ev_grid_coords + flow_ev2d_pred)

            # fuse correlation
            if self.cfgs.fuse_corr:
                if self.cfgs.event_fusion:
                    corr2d, corr3d = self.clfm_corr(uv1, corr2d, corr3d, corr2d_ev)
                else:
                    corr2d, corr3d = self.clfm_corr(uv1, corr2d, corr3d)

            # motion features
            motion_feat2d = self.branch_2d.motion_encoder(flow_2d_pred, corr2d)
            motion_feat3d = self.branch_3d.motion_encoder(xyz1, flow_3d_pred, corr3d, knn_indices=knn_indices)
            motion_evfeat2d = self.branch_2d_event.motion_encoder(flow_ev2d_pred, corr2d_ev)

            # fuse motion features
            if self.cfgs.fuse_motion:
                if self.cfgs.event_fusion:
                    motion_feat2d, motion_feat3d = self.clfm_motion(uv1, motion_feat2d, motion_feat3d, motion_evfeat2d)
                else:
                    motion_feat2d, motion_feat3d = self.clfm_motion(uv1, motion_feat2d, motion_feat3d)

            # 用于KL散度约束
            if self.cfgs.kl_alignment:
                self.motion_feats_rgb.append(motion_feat2d)
                self.motion_feats_event.append(motion_evfeat2d)
                # 3d->2d
                motion_feat3d_interp = self.clfm_motion.interp(uv1, motion_feat2d.float().detach(), motion_feat3d.float().detach())
                self.motion_feats_lidar.append(motion_feat3d_interp)

            # GRU
            h_2d = self.branch_2d.gru(h=h_2d, x=torch.cat([x_2d, motion_feat2d], dim=1))
            h_3d = self.branch_3d.gru(xyz1, h=h_3d, x=torch.cat([x_3d, motion_feat3d], dim=1), knn_indices=knn_indices)
            h_ev2d = self.branch_2d_event.gru(h=h_ev2d, x=torch.cat([x_ev2d, motion_evfeat2d], dim=1))

            # fuse hidden status
            if self.cfgs.fuse_hidden:
                if self.cfgs.event_fusion:
                    h_2d, h_3d = self.clfm_hidden(uv1, h_2d, h_3d, h_ev2d)
                else:
                    h_2d, h_3d = self.clfm_hidden(uv1, h_2d, h_3d)

            # flow head 2D
            delta_flow_2d = self.branch_2d.flow_head(h_2d)
            flow_2d_pred = flow_2d_pred + delta_flow_2d
            flow_2d_up = self.branch_2d.convex_upsampler(h_2d, flow_2d_pred)
            flow_2d_preds.append(flow_2d_up)
            

            # flow head 3D
            delta_flow_3d = self.branch_3d.flow_head(xyz1, h_3d, knn_indices)
            flow_3d_pred = flow_3d_pred + delta_flow_3d
            flow_3d_up = knn_interpolation(xyz1, flow_3d_pred, pc1, k=3)
            flow_3d_preds.append(flow_3d_up)

             # flow head 2D event
            delta_flow_ev2d = self.branch_2d_event.flow_head(h_ev2d)
            flow_ev2d_pred = flow_ev2d_pred + delta_flow_ev2d
            flow_ev2d_up = self.branch_2d_event.convex_upsampler(h_ev2d, flow_ev2d_pred)
            flow_ev2d_preds.append(flow_ev2d_up)

        if self.cfgs.cv_densefication:
            paired_event_slices = input_data['paired_event_slices']
            slice_flow_ev2d_preds = []
            for i in range(len(paired_event_slices)):
                slice_evfeat1_2d = self.branch_2d_event.fnet(paired_event_slices[i][0])
                slice_evfeat2_2d = self.branch_2d_event.fnet(paired_event_slices[i][1])
                slice_evfeatc_2d = self.branch_2d_event.cnet(paired_event_slices[i][0])

                ev_grid_coords = mesh_grid(bs, image_h//8, image_w//8, device=event_voxel1.device)
                flow_ev2d_pred = torch.zeros_like(ev_grid_coords) 
                self.branch_2d_event.correlation.build_cost_volume_pyramid(slice_evfeat1_2d, slice_evfeat2_2d)

                # init hidden state (2D event)
                slice_evfeatc_2d = self.branch_2d_event.cnet_aligner(slice_evfeatc_2d)
                h_ev2d, x_ev2d = torch.split(slice_evfeatc_2d, [128, 128], dim=1)
                h_ev2d = torch.tanh(h_ev2d)
                x_ev2d = torch.relu(x_ev2d)

                flow_ev2d_up = None
                for it in range(n_iters):
                    if it > 0:
                        flow_ev2d_pred = flow_ev2d_pred.detach()

                    # correlation
                    corr2d_ev = self.branch_2d_event.correlation(ev_grid_coords + flow_ev2d_pred)

                    motion_evfeat2d = self.branch_2d_event.motion_encoder(flow_ev2d_pred, corr2d_ev)

                    h_ev2d = self.branch_2d_event.gru(h=h_ev2d, x=torch.cat([x_ev2d, motion_evfeat2d], dim=1))

                    # flow head 2D event
                    delta_flow_ev2d = self.branch_2d_event.flow_head(h_ev2d)
                    flow_ev2d_pred = flow_ev2d_pred + delta_flow_ev2d
                    flow_ev2d_up = self.branch_2d_event.convex_upsampler(h_ev2d, flow_ev2d_pred)

                slice_flow_ev2d_preds.append(flow_ev2d_up)
                
            return flow_2d_preds, flow_3d_preds, flow_ev2d_preds, slice_flow_ev2d_preds
                    

        return flow_2d_preds, flow_3d_preds, flow_ev2d_preds
    

    def get_motion_feats(self):
        return self.motion_feats_rgb, self.motion_feats_event, self.motion_feats_lidar
