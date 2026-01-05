import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
import cv2
import yaml
import utils
import open3d
import logging
import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
from matplotlib.colors import hsv_to_rgb
from omegaconf import DictConfig
from factory import model_factory
from utils import copy_to_device, load_fpm, disp2pc
import pandas as pd
from event_utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from event_utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from event_utils.timers import Timer
from models.event_model.model import *
import time
from image_reconstructor import ImageReconstructor
from typing import Dict, Tuple
import hdf5plugin
import h5py
import math
import torchvision.transforms as transforms
def viz_optical_flow(flow, max_flow=512):
    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)

    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)

    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)

    return image_rgb


class Demo:
    def __init__(self, device: torch.device, cfgs: DictConfig, ckpt_path: str):
        self.cfgs = cfgs
        self.device = device

        logging.info('Creating model: %s' % cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

    def prepare_images_and_depths(self):
        # load images
        image1 = cv2.imread(args.image1)[..., ::-1]
        image2 = cv2.imread(args.image2)[..., ::-1]

        # load disparity maps
        disp1 = -load_fpm(args.disp1)
        disp2 = -load_fpm(args.disp2)

        # lift disparity maps into point clouds
        pc1 = disp2pc(disp1, args.baseline, args.f, args.cx, args.cy)
        pc2 = disp2pc(disp2, args.baseline, args.f, args.cx, args.cy)

        # apply depth mask
        mask1 = (pc1[..., -1] < args.max_depth)
        mask2 = (pc2[..., -1] < args.max_depth)
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # NaN check
        mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1)))
        mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=min(args.n_points, pc1.shape[0]), replace=False)
        indices2 = np.random.choice(pc2.shape[0], size=min(args.n_points, pc2.shape[0]), replace=False)
        pc1, pc2 = pc1[indices1], pc2[indices2]

        return image1, image2, pc1, pc2

    @torch.no_grad()
    def run(self):
        logging.info('Running demo...')
        self.model.eval()

        image1, image2, pc1, pc2 = self.prepare_images_and_depths()

        # numpy -> torch
        images = np.concatenate([image1, image2], axis=-1).transpose([2, 0, 1])
        images = torch.from_numpy(images).float().unsqueeze(0)
        pcs = np.concatenate([pc1, pc2], axis=1).transpose()
        pcs = torch.from_numpy(pcs).float().unsqueeze(0)
        intrinsics = torch.as_tensor([args.f, args.cx, args.cy]).unsqueeze(0)
        
        # inference
        inputs = {'images': images, 'pcs': pcs, 'intrinsics': intrinsics}
        inputs = copy_to_device(inputs, self.device)
        outputs = self.model(inputs)

        # NCHW -> NHWC
        flow_2d = outputs['flow_2d'][0].cpu().numpy().transpose(1, 2, 0)
        flow_3d = outputs['flow_3d'][0].cpu().numpy().transpose()

        self.display(image1, image2, pc1, pc2, flow_2d, flow_3d)

    def display(self, image1, image2, pc1, pc2, flow_2d, flow_3d):
        # visualize optical flow
        flow_2d_img = viz_optical_flow(flow_2d)
        images = np.concatenate([image1, image2, flow_2d_img], axis=0)
        images = cv2.resize(images, dsize=None, fx=0.5, fy=0.5)
        print("11111111")
        # cv2.imshow('', images[..., ::-1])
        cv2.imwrite("0.png", images[..., ::-1])
        
        print("2222222")
        # cv2.waitKey(0)

        # visualize scene flow
        point_cloud1 = open3d.geometry.PointCloud()
        point_cloud2 = open3d.geometry.PointCloud()
        point_cloud3 = open3d.geometry.PointCloud()  # pc1 + flow3d
        point_cloud1.points = open3d.utility.Vector3dVector(pc1)
        point_cloud2.points = open3d.utility.Vector3dVector(pc2)
        point_cloud3.points = open3d.utility.Vector3dVector(pc1 + flow_3d)
        point_cloud1.colors = open3d.utility.Vector3dVector(np.zeros_like(pc1) + [1, 0, 0])
        point_cloud2.colors = open3d.utility.Vector3dVector(np.zeros_like(pc2) + [0, 1, 0])
        point_cloud3.colors = open3d.utility.Vector3dVector(np.zeros_like(pc1) + [0, 0, 1])
        print("33333")
        # 存储为点云形式
        open3d.io.write_point_cloud("3d.pcd", point_cloud3)
        open3d.visualization.draw_geometries([point_cloud1, point_cloud2, point_cloud3])



class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset
        # print(self.t_offset)
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms
    @staticmethod
    # @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        # print(time_ms)
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


# reconstruction
def set_inference_options(parser):

    parser.add_argument('-o', '--output_folder', default=None, type=str)  # if None, will not write the images to disk
    parser.add_argument('--dataset_name', default='reconstruction', type=str)

    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=True)

    """ Display """
    parser.add_argument('--display', dest='display', action='store_true')
    parser.set_defaults(display=False)

    parser.add_argument('--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=False)

    parser.add_argument('--event_display_mode', default='red-blue', type=str,
                        help="Event display mode ('red-blue' or 'grayscale')")

    parser.add_argument('--num_bins_to_show', default=-1, type=int,
                        help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins).")

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    """ Post-processing / filtering """

    # (optional) path to a text file containing the locations of hot pixels to ignore
    parser.add_argument('--hot_pixels_file', default=None, type=str)

    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.set_defaults(flip=False)

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")

    """ Perform color reconstruction? (only use this flag with the DAVIS346color) """
    parser.add_argument('--color', dest='color', action='store_true')
    parser.set_defaults(color=False)

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')
    parser.set_defaults(no_normalize=False)

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')
    parser.set_defaults(no_recurrent=True)


# hdr
from models.hdr_model.video_model import VideoModel
from hdr_utils.utils import *
def save_image(visuals, im_h_pad, im_w_pad, savedir, file_path):
    (filepath, filename) = os.path.split(file_path)
    (name, extension) = os.path.splitext(filename)
    for label, image in visuals.items():
        height_pad = im_h_pad * 2
        width_pad = im_w_pad * 2
        image = delPadding(image, height_pad, -height_pad, width_pad, -width_pad)
        if(label.endswith('output_hdr_rgb')):
            image_numpy = tensor2hdr(image)
            exr_path = os.path.join(savedir, '%s_%s.exr' % (name, label))
            writeEXR(image_numpy, exr_path)

def white_balance(save_dir):
    orig_dir = "%s/exr_results/" % (save_dir)
    files = sorted(os.listdir(orig_dir))
    for i, file in enumerate(files):
        hdr_img = readEXR(os.path.join(orig_dir, file))[:,:,::-1]
        if hdr_img.min() < 0:
            hdr_img = hdr_img - hdr_img.min()
        # white balabce
        if i == 0:
            r_max = hdr_img[:,:,0].max()
            g_max = hdr_img[:,:,1].max()
            b_max = hdr_img[:,:,2].max()
            mat = [[g_max/r_max, 0, 0], [0, 1.0, 0], [0,0,g_max/b_max]]
        hdr_img_wb = whiteBalance_mat(hdr_img, mat)
        writeEXR(hdr_img_wb[:,:,::-1], os.path.join(orig_dir, file))


def __scale(img, target_shorter):
    oh, ow = img.shape[:2]
    if ow >= oh:
        h = target_shorter
        w = int(target_shorter * ow / oh)
    else:
        w = target_shorter
        h = int(target_shorter * oh / ow)
    img_scaled = cv2.resize(img, (w, h), cv2.INTER_LINEAR)

    return img_scaled


if __name__ == '__main__':
    ########################## demo 1: Scene Flow
    parser = argparse.ArgumentParser()

    # model weights
    parser.add_argument('--model', required=True, help='Model name, e.g. vismoflow')
    parser.add_argument('--weights', required=True, help='Path to pretrained weights')

    # HDR RGB input
    parser.add_argument('--image1', required=False, default='img/demo_image1.png')
    parser.add_argument('--image2', required=False, default='img/demo_image2.png')
    
    # inpainted disparity input
    parser.add_argument('--disp1', required=False, default='img/demo_disp1.pfm')
    parser.add_argument('--disp2', required=False, default='img/demo_disp2.pfm')
    
    # disparity -> point clouds
    parser.add_argument('--n_points', required=False, default=8192)
    parser.add_argument('--max_depth', required=False, default=35.0)
    
    # camera intrinsics
    parser.add_argument('--baseline', required=False, default=1.0)
    parser.add_argument('--f', required=False, default=1050.0)
    parser.add_argument('--cx', required=False, default=479.5)
    parser.add_argument('--cy', required=False, default=269.5)
    
    args = parser.parse_args()

    assert args.model in ['camlipwc', 'camliraft']

    with open('conf/model/%s.yaml' % args.model, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    utils.init_logging()

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    else:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))

    demo = Demo(device, cfgs, args.weights)
    demo.run()



    ###################################### demo 2: hdr imageing
    # parser = argparse.ArgumentParser()

    # # basic parameters
    # parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    # parser.add_argument('--name', type=str, default='vidar_video', help='name of the experiment. It decides where to store samples and models')
    # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint', help='models are saved here')
    
    # # model parameters
    # parser.add_argument('--im_type', type=str, default='event', help='choose which neuromorphic data to use as input [event | spike].')
    # parser.add_argument('--model', type=str, default='test', help='chooses which model to use.')
    # parser.add_argument('--netColor', type=str, default='image', help='specify colornet architecture [image | video]')
    # parser.add_argument('--colornet_n_blocks', type=int, default=5, help='number of layers in chrominance compensation network.')
    # parser.add_argument('--state_nc', type=int, default=16, help='number of channels of hidden state in chrominance compensation network.')
    # parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    # parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    # parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    # parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    # parser.add_argument('--up_scale', type=int, default=2, help='upsample scale of upsampling network.')
    
    # # dataset parameters
    # parser.add_argument('--dataset_mode', type=str, default='test', help='chooses how datasets are loaded.')
    # parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    # parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    # parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    # parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    # parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    
    # # additional parameters
    # parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    # parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    # parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    # parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

    # # parser = BaseOptions.initialize(self, parser)  # define shared options
    # parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    # parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    # parser.add_argument('--phase', type=str, default='infer', help='train, infer, etc')
    # parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    # parser.add_argument('--num_test', type=int, default=1000, help='how many test images to run')
    
    # parser.add_argument('--isTrain', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

    # parser.set_defaults(model='test')
    # parser.set_defaults(load_size=parser.get_default('crop_size'))
    # # parser.gpu_ids = 0
    # args = parser.parse_args()
    # parser.isTrain = False
    # parser.num_threads = 0   # test code only supports num_threads = 1
    # parser.batch_size = 1    # test code only supports batch_size = 1
    # parser.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # parser.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # str_ids = args.gpu_ids.split(',')
    # args.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         args.gpu_ids.append(id)
    # # str_ids = args.gpu_ids.split(',')
    # # print(parser.checkpoints_dir)
    # model = VideoModel(args)      # create a model given opt.model and other options
    # model.setup(args)               # regular setup: load and print networks; create schedulers

    # model.eval()
    

    # ldr_files = os.listdir('/mnt/data/zhouhanyu/CVPR2024/Reference/NeurImg-HDR-main/datasets/event_video_dataset/LDR/infer')
    # for i in range(len(ldr_files)):
    #     print(ldr_files[i])
    #     ldr_name = '/mnt/data/zhouhanyu/CVPR2024/Reference/NeurImg-HDR-main/datasets/event_video_dataset/LDR/infer/' + ldr_files[i]
    #     event_name = '/mnt/data/zhouhanyu/CVPR2024/Reference/NeurImg-HDR-main/datasets/event_video_dataset/IM/infer/' + ldr_files[i][:-4] + '.png'
    #     # print(event_name)

    #     im = cv2.imread(event_name, cv2.IMREAD_GRAYSCALE)
    #     im = (im / 255.0).astype(np.float32)

    #     orig_h, orig_w = im.shape[:2]
    #     im_h_pad, im_w_pad = 0, 0
    #     pad_flag = True
    #     if (orig_w % 32 == 0) and (orig_h % 32 == 0):
    #         pad_flag == False
    #     else:
    #         pad_flag == True

    #     if orig_h & 1 != 0:
    #         im = np.pad(im, ((0, 1), (0, 0)), mode='reflect') 
    #     if orig_w & 1 != 0:
    #         im = np.pad(im, ((0, 0), (0, 1)), mode='reflect') 
    #     im_h, im_w = im.shape[:2]
    #     if pad_flag:
    #         if im_h % 32 == 0:
    #             im_h_pad = 0
    #         else:
    #             im_h_pad = int(((im_h//32 + 1)*32-im_h)/2)
    #         if im_w % 32 == 0:
    #             im_w_pad = 0
    #         else:
    #             im_w_pad = int(((im_w//32 + 1)*32-im_w)/2)
    #         im_crop = np.pad(im, ((im_h_pad, im_h_pad), (im_w_pad, im_w_pad)), mode='reflect') 
    #     else:
    #         im_crop = __scale(im, args.resolution//2)
    #         im_crop = cv2.resize(im_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            
    #     im_crop = (im_crop * 2.0 - 1.0).astype(np.float32)
    #     im_tensor = transforms.ToTensor()(im_crop).unsqueeze(0)

    #     # ----------- Load LDR Image -------------
    #     ldr_img = Image.open(ldr_name).convert('RGB')
    #     ldr_img = np.array(ldr_img).astype(np.float32)

    #     if pad_flag:
    #         ldr_h_pad = im_h_pad * args.up_scale
    #         ldr_w_pad = im_w_pad * args.up_scale
    #         ldr_crop = np.pad(ldr_img, ((ldr_h_pad, ldr_h_pad), (ldr_w_pad, ldr_w_pad), (0, 0)), mode='reflect')
    #     else:
    #         ldr_crop = ldr_img
        
    #     ldr_crop = ldr_crop / 255.0
    #     # ldr_crop = ((ldr_crop)**2.2)
    #     ldr_norm = (ldr_crop*2.0-1.0).astype(np.float32)
    #     ldr_rgb = transforms.ToTensor()(ldr_norm).unsqueeze(0)

    #     ldr_yuv = cv2.cvtColor(ldr_crop, cv2.COLOR_RGB2YUV)
    #     ldr_y = ldr_yuv[:,:,0] # [0.0, 1.0]
    #     ldr_u = ldr_yuv[:,:,1] # [-0.5, 0.5]
    #     ldr_v = ldr_yuv[:,:,2] # [-0.5, 0.5]
    #     ldr_y = ldr_y * 2.0 - 1.0
    #     data_ldr_y = transforms.ToTensor()(ldr_y.astype(np.float32)).unsqueeze(0)
        
    #     data_ldr_u = transforms.ToTensor()(ldr_u).unsqueeze(0)
    #     data_ldr_v = transforms.ToTensor()(ldr_v).unsqueeze(0)
    #     # data_ldr_y = data_ldr_y.unsqueeze(0)
    #     input_data = {'input_ldr_y': data_ldr_y, 'input_ldr_u': data_ldr_u, 'input_ldr_v': data_ldr_v, 'input_ldr_rgb': ldr_rgb, 'input_im': im_tensor, 'paths': ldr_name,
    #             'im_h_pad': im_h_pad, 'im_w_pad': im_w_pad}
    #     print(input_data['input_ldr_y'].shape)
    #     model.set_testvideo_input(input_data) # test for video frames
    #     model.test()
    #     visuals = model.get_current_visuals()
    #     img_dir = './outputs/hdr/'
    #     save_image(visuals, input_data['im_h_pad'], input_data['im_w_pad'], img_dir, input_data['paths'])


    ################################ demo 3: event reconstruction
    # parser = argparse.ArgumentParser(
    #     description='Evaluating a trained network')
    # parser.add_argument('-c', '--path_to_model', required=True, type=str,
    #                     help='path to model weights')
    # parser.add_argument('-i', '--input_file', required=True, type=str)
    # parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    # parser.set_defaults(fixed_duration=False)
    # parser.add_argument('-N', '--window_size', default=None, type=int,
    #                     help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    # parser.add_argument('-T', '--window_duration', default=33.33, type=float,
    #                     help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    # parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
    #                     help='in case N (window size) is not specified, it will be \
    #                           automatically computed as N = width * height * num_events_per_pixel')
    # parser.add_argument('--skipevents', default=0, type=int)
    # parser.add_argument('--suboffset', default=0, type=int)
    # parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    # parser.set_defaults(compute_voxel_grid_on_cpu=False)

    # set_inference_options(parser)

    # args = parser.parse_args()

    # # Read sensor size from the first first line of the event file
    # path_to_events = args.input_file

    # width = 640
    # height = 480
    # print('Sensor size: {} x {}'.format(width, height))

    # # Load model
    # # model = load_model(args.path_to_model)
    # raw_model = torch.load(args.path_to_model)
    # arch = raw_model['arch']
    # try:
    #     model_type = raw_model['model']
    # except KeyError:
    #     model_type = raw_model['config']['model']

    # model = eval(arch)(model_type)
    # model.load_state_dict(raw_model['state_dict'])

    # # device = get_device(args.use_gpu)
    # device = torch.device('cuda:0')

    # model = model.to(device)
    # model.eval()

    # reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    # """ Read chunks of events using Pandas """

    # # Loop through the events and reconstruct images
    # N = args.window_size
    # if not args.fixed_duration:
    #     if N is None:
    #         N = int(width * height * args.num_events_per_pixel)
    #         print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
    #             N, args.num_events_per_pixel))
    #     else:
    #         print('Will use {} events per tensor (user-specified)'.format(N))
    #         mean_num_events_per_pixel = float(N) / float(width * height)
    #         if mean_num_events_per_pixel < 0.1:
    #             print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
    #                 The reconstruction results might be suboptimal.'.format(N))
    #         elif mean_num_events_per_pixel > 1.5:
    #             print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
    #                 The reconstruction results might be suboptimal.'.format(N))

    # initial_offset = args.skipevents
    # sub_offset = args.suboffset
    # start_index = initial_offset + sub_offset

    # if args.compute_voxel_grid_on_cpu:
    #     print('Will compute voxel grid on CPU.')

    # # if args.fixed_duration:
    # #     event_window_iterator = FixedDurationEventReader(path_to_events,
    # #                                                      duration_ms=args.window_duration,
    # #                                                      start_index=start_index)
    # # else:
    # #     event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)
    # ev_data_file = '/mnt/data/zhouhanyu/NIPS2023/dataset/DSEC/train/zurich_city_11_c/events/events.h5'
    # ev_location = h5py.File(str(ev_data_file), 'r')
    # events_slicer = EventSlicer(ev_location)
    # time_delta = 20000
    # with Timer('Processing entire dataset'):
    #     with open('/mnt/data/zhouhanyu/NIPS2023/dataset/DSEC/train/zurich_city_11_c/timestamps.txt', 'r') as file:
    #         lines = file.readlines()
    #         for line in lines:
    #         # for idx in range(len(100))
    #             # frame_name = str(frame_id).zfill(6)
    #             timestamp = int(line)
    #             print(line)
    #             events = events_slicer.get_events(timestamp, timestamp+time_delta)

    #             p = events['p'].astype(np.int8)
    #             t = events['t'].astype(np.float64)
    #             x = events['x']
    #             y = events['y']
    #             p = 2*p - 1

    #             events_rectified = np.stack([t, x, y, p], axis=-1)
    #             last_timestamp = events_rectified[-1, 0]
    #             print(events_rectified.shape)
    #             event_tensor = events_to_voxel_grid_pytorch(events_rectified,
    #                                                             num_bins=model.num_bins,
    #                                                             width=width,
    #                                                             height=height,
    #                                                             device=device)
    #             # print(event_tensor.shape)
    #             num_events_in_window = events_rectified.shape[0]
    #             reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
    #             start_index += num_events_in_window





