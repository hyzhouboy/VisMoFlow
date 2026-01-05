import torch
from torch.distributions import Normal, Independent, kl
from .utils import resize_flow2d, batch_indexing


def calc_pyramid_loss_2d(flows, target, cfgs):
    """Pyramid loss for PWC-Net."""

    assert len(flows) <= len(cfgs.level_weights)

    total_loss = 0
    for pred, level_weight in zip(flows, cfgs.level_weights):
        assert pred.shape[1] == 2  # [B, 2, H, W]

        if target.shape[1] == 3:
            flow_mask = target[:, 2] > 0
        else:
            flow_mask = torch.ones_like(target)[:, 0] > 0

        diff = torch.abs(resize_flow2d(pred, target.shape[2], target.shape[3]) - target[:, :2])

        if cfgs.order == 'robust':
            loss_l1_map = torch.pow(diff.sum(dim=1) + 0.01, 0.4)
            loss_l1 = loss_l1_map[flow_mask].mean()
            total_loss += level_weight * loss_l1
        elif cfgs.order == 'l2-norm':
            loss_l2_map = torch.linalg.norm(diff, dim=1)
            loss_l2 = loss_l2_map[flow_mask].mean()
            total_loss += level_weight * loss_l2
        else:
            raise NotImplementedError

    return total_loss


def calc_pyramid_loss_3d(flows, target, cfgs, indices):
    """Pyramid loss for PointPWC-Net."""

    assert len(flows) <= len(cfgs.level_weights)

    total_loss = 0
    for idx, (flow, level_weight) in enumerate(zip(flows, cfgs.level_weights)):
        level_target = batch_indexing(target, indices[idx])

        if level_target.shape[1] == 4:
            flow_mask = level_target[:, 3, :] > 0
            diff = flow - level_target[:, :3, :]
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
            epe_l2 = torch.linalg.norm(diff, dim=1)[flow_mask].mean()
        else:
            diff = flow - level_target
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4).mean()
            epe_l2 = torch.linalg.norm(diff, dim=1).mean()

        if cfgs.order == 'robust':
            total_loss += level_weight * epe_l1
        elif cfgs.order == 'l2-norm':
            total_loss += level_weight * epe_l2
        else:
            raise NotImplementedError

    return total_loss


def calc_sequence_loss_2d(flow_preds, target, cfgs):
    """Sequence loss for RAFT."""

    n_preds = len(flow_preds)
    total_loss = 0

    if target.shape[1] == 3:
        flow_mask = target[:, 2] > 0
    else:
        flow_mask = torch.ones_like(target)[:, 0] > 0

    for i in range(n_preds):
        diff = flow_preds[i] - target[:, :2]

        if cfgs.order == 'l2-norm':
            loss = torch.linalg.norm(diff, dim=1)[flow_mask].mean()
        elif cfgs.order == 'l1':
            loss = torch.sum(diff.abs(), dim=1)[flow_mask].mean()
        elif cfgs.order == 'robust':
            loss = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
        else:
            raise ValueError

        weight = cfgs.gamma ** (n_preds - i - 1)
        total_loss += weight * loss

    return total_loss


def calc_kl_divergence(feat_rgb, feat_event, feat_lidar):
    _, _, H, W = feat_rgb.shape

    tanh_feat_rgb = torch.tanh(feat_rgb)
    tanh_feat_event = torch.tanh(feat_event)
    tanh_feat_lidar = torch.tanh(feat_lidar)
    
    dist_rgb = Independent(Normal(loc=tanh_feat_rgb, scale=torch.exp(tanh_feat_rgb)), 1)
    dist_event = Independent(Normal(loc=tanh_feat_event, scale=torch.exp(tanh_feat_event)), 1)
    dist_lidar = Independent(Normal(loc=tanh_feat_lidar, scale=torch.exp(tanh_feat_lidar)), 1)
    
    bi_kl_rgb_event = torch.mean(kl.kl_divergence(dist_rgb, dist_event)) + torch.mean(kl.kl_divergence(dist_event, dist_rgb))
    bi_kl_rgb_lidar = torch.mean(kl.kl_divergence(dist_rgb, dist_lidar)) + torch.mean(kl.kl_divergence(dist_lidar, dist_rgb))
    bi_kl_event_lidar = torch.mean(kl.kl_divergence(dist_event, dist_lidar)) + torch.mean(kl.kl_divergence(dist_lidar, dist_event))

    kl_div = -1.0 * (bi_kl_rgb_event + bi_kl_rgb_lidar + bi_kl_event_lidar)
    
    kl_div /= H * W

    return kl_div

def calc_sequence_loss_3d(flow_preds, target, cfgs):
    """Sequence loss for Point-RAFT."""
    
    n_preds = len(flow_preds)
    total_loss = 0

    if target.shape[1] == 4:
        flow_mask = target[:, 3] > 0
    else:
        flow_mask = torch.ones_like(target)[:, 0] > 0

    for i in range(n_preds):
        diff = flow_preds[i] - target[:, :3]

        if cfgs.order == 'l2-norm':
            loss = torch.linalg.norm(diff, dim=1)[flow_mask].mean()
        elif cfgs.order == 'l1':
            loss = torch.sum(diff.abs(), dim=1)[flow_mask].mean()
        elif cfgs.order == 'robust':
            loss = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
        else:
            raise ValueError

        weight = cfgs.gamma ** (n_preds - i - 1)
        total_loss += weight * loss

    return total_loss



def calc_slice_loss_census(img1, img2, flow):
    weight = 0.5
    # compute grid
    h, w = img1.size(2), img1.size(3)
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h))

    grid_x = grid_x.type_as(img1).transpose(0, 1).unsqueeze(0)
    grid_y = grid_y.type_as(img1).transpose(0, 1).unsqueeze(0)
    
    grid_x = grid_x + flow[:, 0, :, :]  # 对应光流的x位移
    grid_y = grid_y + flow[:, 1, :, :]  # 对应光流的y位移
    
    grid = torch.stack((grid_x, grid_y), dim=3)

    # 双线性插值
    grid[:, :, :, 0] = 2 * grid[:, :, :, 0] / (w - 1) - 1
    grid[:, :, :, 1] = 2 * grid[:, :, :, 1] / (h - 1) - 1
    warped_image = torch.nn.functional.grid_sample(img1, grid, mode='bilinear', padding_mode='border')

    photometric_loss = torch.abs(img2 - warped_image).mean()
    photometric_loss = weight * photometric_loss

    return photometric_loss


