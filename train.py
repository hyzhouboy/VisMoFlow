import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import glob
import time
import utils
import hydra
import random
import shutil
import logging
import numpy as np
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from factory import dataset_factory, model_factory, optimizer_factory
from utils import copy_to_device, override_cfgs, FastDataLoader, get_max_memory, get_grad_norm
from models.utils import timer
from image_reconstructor import ImageReconstructor
import hdf5plugin
import h5py
import math
import torchvision.transforms as transforms
from event_utils.event_readers import EventSlicer
from event_utils.inference_utils import events_to_voxel_grid_pytorch
from event_utils.timers import Timer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, OneCycleLR
from models.hdr_model.video_model import VideoModel
from hdr_utils.utils import *
from utils import Sobel, events_to_event_image, disp2pc, depth2pc
import imageio
import open3d
from skimage import io
# torch.cuda.set_device(0)
#在清晰KITTI图像上训练图像-激光雷达融合场景流
class Trainer_training_flow:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'), self.cfgs.debug)

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                init_process_group('nccl', 'tcp://localhost:%d' % self.cfgs.port,
                                   world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.trainset.n_workers = int(self.cfgs.trainset.n_workers / self.n_gpus)
                self.cfgs.valset.n_workers = int(self.cfgs.valset.n_workers / self.n_gpus)
            if not cfgs.debug:
                cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info('Logs will be saved to %s' % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            logging.info('Configurations:\n' + OmegaConf.to_yaml(self.cfgs))
        else:
            logging.root.disabled = True

        logging.info('Loading training set from %s' % self.cfgs.trainset.root_dir)
        self.train_dataset = dataset_factory(self.cfgs.trainset)
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
        )

        logging.info('Loading validation set from %s' % self.cfgs.valset.root_dir)
        self.val_dataset = dataset_factory(self.cfgs.valset)
        self.val_sampler = DistributedSampler(self.val_dataset) if self.n_gpus > 1 else None
        self.val_loader = FastDataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=False,
            num_workers=self.cfgs.valset.n_workers,
            pin_memory=True,
            sampler=self.val_sampler,
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model)
        self.model.to(device=self.device)

        n_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))

        if self.n_gpus > 1:
            if self.cfgs.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.opt)
        self.optimizer, self.scheduler = optimizer_factory(self.cfgs.training, self.model)
        self.scheduler.step(self.curr_epoch - 1)

        self.amp_scaler = GradScaler(enabled=self.cfgs.amp)

    def run(self):
        while self.curr_epoch <= self.cfgs.training.epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()

            # if self.curr_epoch % self.cfgs.val_interval == 0:
                # self.validate()
            if self.curr_epoch % self.cfgs.training.saved_interval == 0:
                self.save_ckpt()

            self.scheduler.step(self.curr_epoch)

            self.curr_epoch += 1

    def train_one_epoch(self):
        logging.info('Start training...')

        self.ddp.train()
        self.model.clear_metrics()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        self.save_scalar_summary({'learning_rate': lr}, prefix='train')

        start_time = time.time()
        for i, inputs in enumerate(self.train_loader):
            inputs = copy_to_device(inputs, self.device)

            # forward
            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                self.ddp.forward(inputs)
                loss = self.model.get_loss()

            # backward
            self.amp_scaler.scale(loss).backward()

            # get grad norm statistics
            grad_norm_2d = get_grad_norm(self.model, prefix='core.branch_2d')
            grad_norm_3d = get_grad_norm(self.model, prefix='core.branch_3d')
            self.model.update_metrics('grad_norm_2d', grad_norm_2d)
            self.model.update_metrics('grad_norm_3d', grad_norm_3d)

            # grad clip
            if 'grad_max_norm' in self.cfgs.training.keys():
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfgs.training.grad_max_norm
                )

            # update
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

            timing = time.time() - start_time
            start_time = time.time()
            mem = get_max_memory(self.device, self.n_gpus)

            logging.info('Epoch: [%d/%d]' % (self.curr_epoch, self.cfgs.training.epochs) +
                            '[%d/%d] ' % (i + 1, len(self.train_loader)) +
                            'loss: %.1f, time: %.2fs, mem: %dM' % (loss, timing, mem))

            for k, v in timer.get_timing_stat().items():
                logging.info('Function "%s" takes %.1fms' % (k, v))

            timer.clear_timing_stat()

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='train')

    @torch.no_grad()
    def validate(self):
        logging.info('Start validating...')

        self.ddp.eval()
        self.model.clear_metrics()

        for inputs in tqdm(self.val_loader):
            inputs = copy_to_device(inputs, self.device)
            self.ddp.forward(inputs)

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='val')

        for k, v in metrics.items():
            logging.info('%s: %.4f' % (k, v))

        if self.model.is_better(metrics, self.best_metrics):
            self.best_metrics = metrics
            self.save_ckpt('best.pt')

    def save_scalar_summary(self, scalar_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_scalar_summary:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(
                    prefix + '/' + name,
                    scalar_summary[name],
                    self.curr_epoch
                )

    def save_image_summary(self, image_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_image_summary:
            for name in image_summary.keys():
                self.summary_writer.add_image(
                    prefix + '/' + name,
                    image_summary[name],
                    self.curr_epoch
                )

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        logging.info('Loading checkpoint from %s' % filepath)
        checkpoint = torch.load(filepath, self.device)
        if resume:
            self.curr_epoch = checkpoint['last_epoch'] + 1
            self.best_metrics = checkpoint['best_metrics']
            logging.info('Current best metrics: %s' % str(self.best_metrics))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)



# 生成KITTI数据集对应仿真事件的重建图像
class Trainer_reconstruction:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0

        self.path_to_events = self.cfgs.reconstruction.input_file
        self.width = 1242
        self.height = 376
        # print('Sensor size: {} x {}'.format(width, height))
        logging.info('Sensor size: %d x %d ' % (self.width, self.height))

        # Load model
        raw_model = torch.load(self.cfgs.reconstruction.model_path)
        arch = raw_model['arch']
        try:
            model_type = raw_model['model']
        except KeyError:
            model_type = raw_model['config']['model']
        self.model = eval(arch)(model_type)
        self.model.load_state_dict(raw_model['state_dict'])

        self.device = torch.device('cuda:0')
        self.model = self.model.to(self.device)
        self.model.eval()


        self.reconstructor = ImageReconstructor(self.model, self.height, self.width, self.model.num_bins, self.cfgs.reconstruction)

    def run(self):
        initial_offset = self.cfgs.reconstruction.skipevents
        sub_offset = self.cfgs.reconstruction.suboffset
        start_index = initial_offset + sub_offset

        # inference
        ev_location = h5py.File(self.path_to_events, 'r')
        events_slicer = EventSlicer(ev_location)
        time_delta = self.cfgs.Events.time_delta
        with Timer('Processing entire dataset'):
            with open(str(self.cfgs.reconstruction.timestamps_file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    timestamp = int(line)
                    print(line)
                    events = events_slicer.get_events(timestamp, timestamp+time_delta)

                    p = events['p'].astype(np.int8)
                    t = events['t'].astype(np.float64)
                    x = events['x']
                    y = events['y']
                    p = 2*p - 1

                    events_rectified = np.stack([t, x, y, p], axis=-1)
                    last_timestamp = events_rectified[-1, 0]
                    print(events_rectified.shape)
                    event_tensor = events_to_voxel_grid_pytorch(events_rectified,
                                                                    num_bins=self.model.num_bins,
                                                                    width=self.width,
                                                                    height=self.height,
                                                                    device=self.device)
                    # print(event_tensor.shape)
                    num_events_in_window = events_rectified.shape[0]
                    self.reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
                    start_index += num_events_in_window



# 生成KITTI数据集对应HDR图像
class Trainer_HDR:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'), self.cfgs.debug)

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                init_process_group('nccl', 'tcp://localhost:%d' % self.cfgs.port,
                                   world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.trainset.n_workers = int(self.cfgs.trainset.n_workers / self.n_gpus)
            if not cfgs.debug:
                cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info('Logs will be saved to %s' % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            logging.info('Configurations:\n' + OmegaConf.to_yaml(self.cfgs))
        else:
            logging.root.disabled = True

        logging.info('Loading training set from %s' % self.cfgs.trainset.root_dir)
        self.train_dataset = dataset_factory(self.cfgs.trainset)
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
        )

        # flow 部分
        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model)
        self.model.to(device=self.device)

        n_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))

         
        if self.n_gpus > 1:
            if self.cfgs.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.opt)

        # hdr模型
        self.cfgs.HDR.isTrain = False
        self.cfgs.HDR.num_threads = 4   # test code only supports num_threads = 1
        self.cfgs.HDR.batch_size = 2    # test code only supports batch_size = 1
        self.cfgs.HDR.serial_batches = False  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.cfgs.HDR.no_flip = True    # no flip; comment this line if results on flipped images are needed.        
        self.cfgs.HDR.gpu_ids = []
        for id in range(self.n_gpus):
            if id >= 0:
                self.cfgs.HDR.gpu_ids.append(id)
        
        self.hdr_model = VideoModel(self.cfgs.HDR)      # create a model given opt.model and other options
        self.hdr_model.setup(self.cfgs.HDR)               # regular setup: load and print networks; create schedulers

        self.sobel = Sobel()
        # 优化器
        self.optimizer, self.scheduler = optimizer_factory(self.cfgs.training, self.model)
        self.scheduler.step(self.curr_epoch - 1)
        self.amp_scaler = GradScaler(enabled=self.cfgs.amp)
        # HDR优化器
        # self.optimizer_hdr = torch.optim.AdamW(self.hdr_model.parameters(), lr=0.0015, weight_decay=0.85, eps=1e-8)
        # self.hdr_scheduler = OneCycleLR(self.optimizer_hdr, lr=0.0015, 100000, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    def run(self):
        while self.curr_epoch <= self.cfgs.training.epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()

            if self.curr_epoch % self.cfgs.training.saved_interval == 0:
                self.save_ckpt()


            self.scheduler.step(self.curr_epoch)
            self.hdr_scheduler.step(self.curr_epoch)

            self.curr_epoch += 1

    def train_one_epoch(self):
        logging.info('Start training...')

        self.ddp.train()
        self.model.clear_metrics()
        self.optimizer.zero_grad()

        # hdr网络参数更新
        self.hdr_model.eval()
        # self.optimizer_hdr.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        self.save_scalar_summary({'learning_rate': lr}, prefix='train')

        start_time = time.time()
        # 数据集：原始基础上，加入事件流、重建图像、yuv分解等等
        input_1_hdr = {}
        input_2_hdr = {}
        inputs_sceneflow = {}
        for i, inputs in enumerate(self.train_loader):
            
            input_1_hdr['intensity_image'] = inputs['intensity_image_1']
            input_1_hdr['ldr_y'] = inputs['ldr_1_y']
            input_1_hdr['ldr_u'] = inputs['ldr_1_u']
            input_1_hdr['ldr_v'] = inputs['ldr_1_v']

            input_2_hdr['intensity_image'] = inputs['intensity_image_2']
            input_2_hdr['ldr_y'] = inputs['ldr_2_y']
            input_2_hdr['ldr_u'] = inputs['ldr_2_u']
            input_2_hdr['ldr_v'] = inputs['ldr_2_v']

            input_1_hdr = copy_to_device(input_1_hdr, self.device)
            input_2_hdr = copy_to_device(input_2_hdr, self.device)
            
            # forward
            with torch.no_grad():
                self.hdr_model.forward(input_1_hdr)
                hdr_image_1 = self.hdr_model.get_current_visuals()['output_hdr_rgb']
                self.hdr_model.forward(input_2_hdr)
                hdr_image_2 = self.hdr_model.get_current_visuals()['output_hdr_rgb']
            
            pcs = torch.cat((inputs['pc_1'], inputs['pc_2']), 1)
            images = torch.cat((hdr_image_1, hdr_image_2), 1)
            inputs_sceneflow['index'] = inputs['index']
            inputs_sceneflow['input_h'] = inputs['input_h']
            inputs_sceneflow['input_w'] = inputs['input_w']
            inputs_sceneflow['images'] = images
            inputs_sceneflow['flow_2d'] = inputs['flow_2d']
            inputs_sceneflow['pcs'] = pcs
            inputs_sceneflow['flow_3d'] = inputs['flow_3d']
            inputs_sceneflow['intrinsics'] = inputs['intrinsics']
            inputs_sceneflow = copy_to_device(inputs_sceneflow, self.device)
            
            # images = np.concatenate([image1, image2], axis=-1)
            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                flow = self.ddp.forward(inputs_sceneflow)
                loss_epe = self.model.get_loss()
                optical_flow = flow['flow_2d']
                # scene_flow = flow['flow_3d']
                # 时空梯度一致性损失
                img_gradx, img_grady = self.sobel(inputs['image_1'])
                # warped_img_grady = F.grid_sample(img_grady, grid_pos, mode="bilinear", padding_mode="zeros")
                img_deltaL = img_grady * optical_flow[:, 0:1, :, :] + img_gradx * optical_flow[:, 1:2, :, :]
                event_deltaL = torch.mean(inputs['events_1'], dim=1)
                loss_gradient_consistency = (img_deltaL - event_deltaL).abs() * 0.05
                loss_gradient_consistency = loss_gradient_consistency.mean()
                # loss_weight * torch.sum((event_flow - image_flow)**2, dim=1).sqrt()
                loss = loss_epe + loss_gradient_consistency


            # backward
            self.amp_scaler.scale(loss).backward()

            # get grad norm statistics
            grad_norm_2d = get_grad_norm(self.model, prefix='core.branch_2d')
            grad_norm_3d = get_grad_norm(self.model, prefix='core.branch_3d')
            self.model.update_metrics('grad_norm_2d', grad_norm_2d)
            self.model.update_metrics('grad_norm_3d', grad_norm_3d)

            # grad clip
            if 'grad_max_norm' in self.cfgs.training.keys():
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfgs.training.grad_max_norm
                )

            # update
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

            timing = time.time() - start_time
            start_time = time.time()
            mem = get_max_memory(self.device, self.n_gpus)

            logging.info('Epoch: [%d/%d]' % (self.curr_epoch, self.cfgs.training.epochs) +
                            '[%d/%d] ' % (i + 1, len(self.train_loader)) +
                            'loss: %.1f, time: %.2fs, mem: %dM' % (loss, timing, mem))

            for k, v in timer.get_timing_stat().items():
                logging.info('Function "%s" takes %.1fms' % (k, v))

            timer.clear_timing_stat()

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='train')

    def save_image(visuals, im_h_pad, im_w_pad, savedir, file_path):
        (filepath, filename) = os.path.split(file_path[0])
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

    def save_scalar_summary(self, scalar_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_scalar_summary:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(
                    prefix + '/' + name,
                    scalar_summary[name],
                    self.curr_epoch
                )

    def save_image_summary(self, image_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_image_summary:
            for name in image_summary.keys():
                self.summary_writer.add_image(
                    prefix + '/' + name,
                    image_summary[name],
                    self.curr_epoch
                )

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        logging.info('Loading checkpoint from %s' % filepath)
        checkpoint = torch.load(filepath, self.device)
        if resume:
            self.curr_epoch = checkpoint['last_epoch'] + 1
            self.best_metrics = checkpoint['best_metrics']
            logging.info('Current best metrics: %s' % str(self.best_metrics))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)





# 生成KITTI数据集对应点云与事件的修补
class Trainer_CloudInpaiting:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        self.cfgs = cfgs
        self.root_dir = self.cfgs.CloudInpainting.root_dir
        self.child_name = self.cfgs.CloudInpainting.child_name
        self.width = self.cfgs.CloudInpainting.width
        self.height = self.cfgs.CloudInpainting.height
        self.time_delta = self.cfgs.CloudInpainting.time_delta
        
        self.parent_path = os.path.join(self.root_dir, self.child_name)

        self.baseline = self.cfgs.CloudInpainting.baseline
        self.f = self.cfgs.CloudInpainting.f
        self.cx = self.cfgs.CloudInpainting.cx
        self.cy = self.cfgs.CloudInpainting.cy
        self.max_depth = self.cfgs.CloudInpainting.max_depth
        self.min_depth = self.cfgs.CloudInpainting.min_depth

        # 以DSEC为例
        ev_data_file = os.path.join(self.parent_path, 'events/events.h5')
        ev_location = h5py.File(str(ev_data_file), 'r')
        self.events_slicer = EventSlicer(ev_location)

        self.files = sorted(os.listdir(os.path.join(self.parent_path, 'disparity/event')))
        # 创建文件夹
        if os.path.exists(os.path.join(self.parent_path, 'cluster')) == False:
            os.mkdir(os.path.join(self.parent_path, 'cluster'))
            os.mkdir(os.path.join(self.parent_path, 'cluster/event'))
            os.mkdir(os.path.join(self.parent_path, 'cluster/fused_depth'))
            os.mkdir(os.path.join(self.parent_path, 'cluster/origin_depth'))
        if os.path.exists(os.path.join(self.parent_path, 'enhanced_depth')) == False:
            os.mkdir(os.path.join(self.parent_path, 'enhanced_depth'))
        if os.path.exists(os.path.join(self.parent_path, 'enhanced_point_cloud')) == False:
            os.mkdir(os.path.join(self.parent_path, 'enhanced_point_cloud'))


    def run(self):
        logging.info('Start enhancing point cloud...')

        with open(os.path.join(self.parent_path, 'disparity/timestamps.txt'), 'r') as file:
            lines = file.readlines()
            for index in range(len(self.files)):
                line = lines[index]

                print(self.files[index])
                timestamp = int(line)
                events = self.events_slicer.get_events(timestamp, timestamp+self.time_delta)

                p = events['p'].astype(np.int8)
                t = events['t'].astype(np.float64)
                x = events['x']
                y = events['y']
                p = 2*p - 1

                events_rectified = np.stack([t, x, y, p], axis=-1)
                
                event_image = events_to_event_image(
                    event_sequence=events_rectified,
                    height=480,
                    width=640
                ).numpy()
                save_name = os.path.join(self.parent_path + '/event_frame', self.files[index])
                imageio.imsave(save_name, event_image.transpose(1,2,0))
                

                # save depth
                disp = io.imread(os.path.join(os.path.join(self.parent_path, 'disparity/event'), self.files[index]))
                # pc = disp2pc(disp, baseline=self.baseline, f=self.f, cx=self.cx, cy=self.cy)
                # disp2 = io.imread(os.path.join(os.path.join(path, 'disparity/event'), files[index+1]))
                depth = self.baseline * self.f / (disp + 1e-5)
                mask = (depth[:,:] < self.max_depth)
                depth = depth*mask
                # cv2.imwrite(os.path.join(os.path.join(path, 'depth'), files[index]), (100.0*depth).astype(np.uint16))


                # 聚类---SLIC KNN DBSCAN
                img = cv2.imread(os.path.join(os.path.join(self.parent_path, 'event_frame'), self.files[index]))
                # DBSCAN()
                slic = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLIC, region_size=10, ruler=20.0) 
                slic.iterate(10)     # 迭代次数，越大效果越好
                mask_slic = slic.getLabelContourMask()     # 获取Mask，超像素边缘Mask==1
                label_slic = slic.getLabels()     # 获取超像素标签
                # 邻域构建
                mask_inv_slic = cv2.bitwise_not(mask_slic)  
                img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #在原图上绘制超像素边界

                color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                color_img[:] = (0, 0 , 0)
                result_ = cv2.bitwise_and(color_img, color_img, mask=mask_slic)
                result = cv2.add(img_slic, result_)
                cv2.imwrite(os.path.join(os.path.join(self.parent_path, 'cluster/event'), self.files[index]), result)

                # 深度融合
                filled_image = np.zeros_like(depth)
                mask = (img[:,:, 1] == 0)
                # mask = 
                for label in np.unique(label_slic):
                    y, x = np.where(label_slic == label)
                    # 计算当前区域的深度均值
                    non_zero_depth = depth[y, x][depth[y, x] > 0]
                    if len(non_zero_depth) > 0:
                        mean_depth = np.mean(non_zero_depth)
                        filled_image[y, x] = mean_depth
                # print(filled_image.shape)
                filled_depth = mask * filled_image
                fuse_mask = (depth[:, :] == 0)
                fused_depth = filled_depth * fuse_mask + depth
                cv2.imwrite(os.path.join(os.path.join(self.parent_path, 'enhanced_depth'), self.files[index]), (100.0*fused_depth).astype(np.uint16))

                fused_pc = depth2pc(fused_depth, f=self.f, cx=self.cx, cy=self.cy)
                mask = (fused_pc[..., -1] < self.max_depth)
                fused_pc = fused_pc[mask]
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(fused_pc)
                open3d.io.write_point_cloud(os.path.join(os.path.join(self.parent_path, 'enhanced_point_cloud'), self.files[index][:-4]+'.pcd'), point_cloud)
                # open3d.visualization.draw_geometries([point_cloud])
                # cv2.waitkey(0)

                # color depth
                gray_depth_image_origin = (255 * (depth - self.min_depth) / (self.max_depth - self.min_depth)).astype(np.uint8)
                gray_depth_image_fused = (255 * (fused_depth - self.min_depth) / (self.max_depth - self.min_depth)).astype(np.uint8)
                im_color_origin=cv2.applyColorMap(gray_depth_image_origin,cv2.COLORMAP_BONE)
                im_color_fused=cv2.applyColorMap(gray_depth_image_fused,cv2.COLORMAP_BONE)
                im_ofigin=Image.fromarray(im_color_origin)
                im_fused=Image.fromarray(im_color_fused)
                im_ofigin.save(os.path.join(os.path.join(self.parent_path, 'cluster/origin_depth'), self.files[index]))
                im_fused.save(os.path.join(os.path.join(self.parent_path, 'cluster/fused_depth'), self.files[index]))
                
        


# 运动融合阶段，输入：事件、图像、点云，输出：融合后的场景流和光流
# 运动融合第一阶段：训练事件体素作为输入估计的光流模型
class Trainer_MotionFusion:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'), self.cfgs.debug)

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                init_process_group('nccl', 'tcp://localhost:%d' % self.cfgs.port,
                                   world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.trainset.n_workers = int(self.cfgs.trainset.n_workers / self.n_gpus)
                self.cfgs.valset.n_workers = int(self.cfgs.valset.n_workers / self.n_gpus)
            if not cfgs.debug:
                cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info('Logs will be saved to %s' % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            logging.info('Configurations:\n' + OmegaConf.to_yaml(self.cfgs))
        else:
            logging.root.disabled = True

        logging.info('Loading training set from %s' % self.cfgs.trainset.root_dir)
        self.train_dataset = dataset_factory(self.cfgs.trainset)
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model)
        self.model.to(device=self.device)

        n_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))

        if self.n_gpus > 1:
            if self.cfgs.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.opt)
        self.optimizer, self.scheduler = optimizer_factory(self.cfgs.training, self.model)
        self.scheduler.step(self.curr_epoch - 1)

        self.amp_scaler = GradScaler(enabled=self.cfgs.amp)

    def run(self):
        while self.curr_epoch <= self.cfgs.training.epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            self.train_one_epoch()

            # if self.curr_epoch % self.cfgs.val_interval == 0:
                # self.validate()
            if self.curr_epoch % self.cfgs.training.saved_interval == 0:
                self.save_ckpt()

            self.scheduler.step(self.curr_epoch)

            self.curr_epoch += 1

    def train_one_epoch(self):
        logging.info('Start training...')

        self.ddp.train()
        self.model.clear_metrics()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        self.save_scalar_summary({'learning_rate': lr}, prefix='train')

        start_time = time.time()
        input_data = dict()
        for i, inputs in enumerate(self.train_loader):
            input_data['images'] = torch.cat((inputs['image_1'], inputs['image_2']), dim=1)
            input_data['pcs'] = torch.cat((inputs['pc_1'], inputs['pc_2']), dim=1)
            input_data['voxels'] = torch.cat((inputs['event_slice_voxel'][0], inputs['event_slice_voxel'][-1]), dim=1)
            input_data['slices_voxels'] = inputs['event_slice_voxel']
            input_data['intrinsics'] = inputs['intrinsics']
            input_data['flow_2d'] = inputs['flow_2d']
            input_data['flow_3d'] = inputs['flow_3d']

            input_data = copy_to_device(input_data, self.device)

            # forward
            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                self.ddp.forward(input_data)
                loss = self.model.get_loss()

            # backward
            self.amp_scaler.scale(loss).backward()

            # get grad norm statistics
            grad_norm_2d = get_grad_norm(self.model, prefix='core.branch_2d')
            grad_norm_3d = get_grad_norm(self.model, prefix='core.branch_3d')
            self.model.update_metrics('grad_norm_2d', grad_norm_2d)
            self.model.update_metrics('grad_norm_3d', grad_norm_3d)

            # grad clip
            if 'grad_max_norm' in self.cfgs.training.keys():
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfgs.training.grad_max_norm
                )

            # update
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

            timing = time.time() - start_time
            start_time = time.time()
            mem = get_max_memory(self.device, self.n_gpus)

            logging.info('Epoch: [%d/%d]' % (self.curr_epoch, self.cfgs.training.epochs) +
                            '[%d/%d] ' % (i + 1, len(self.train_loader)) +
                            'loss: %.1f, time: %.2fs, mem: %dM' % (loss, timing, mem))

            for k, v in timer.get_timing_stat().items():
                logging.info('Function "%s" takes %.1fms' % (k, v))

            timer.clear_timing_stat()

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='train')


    def save_scalar_summary(self, scalar_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_scalar_summary:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(
                    prefix + '/' + name,
                    scalar_summary[name],
                    self.curr_epoch
                )

    def save_image_summary(self, image_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_image_summary:
            for name in image_summary.keys():
                self.summary_writer.add_image(
                    prefix + '/' + name,
                    image_summary[name],
                    self.curr_epoch
                )

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        # 临时加载RAFT的代码，如果有全套模型就不要加载部分了
        logging.info('Loading checkpoint from %s' % filepath)
        if self.cfgs.trainset.stage == 4:
            checkpoint = torch.load(filepath,  self.device)
            state_dict = checkpoint['state_dict']
            new_model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
            print(len(pretrained_dict))
            new_model_dict.update(pretrained_dict)
            # print(new_model_dict['core.clfm_motion.interp.score_net.1.conv_fn.weight'])
            self.model.load_state_dict(new_model_dict, strict=True)
        else:
            checkpoint = torch.load(filepath, self.device)
            if resume:
                self.curr_epoch = checkpoint['last_epoch'] + 1
                self.best_metrics = checkpoint['best_metrics']
                logging.info('Current best metrics: %s' % str(self.best_metrics))
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)

    


def create_trainer(device_id, cfgs):
    device = torch.device('cpu' if device_id is None else 'cuda:%d' % device_id)
    if cfgs.trainset.name == 'kitti':
        trainer = Trainer_training_flow(device, cfgs)
        trainer.run()
    elif cfgs.trainset.name == 'kitti_event':
        if cfgs.trainset.stage == 1:
            trainer = Trainer_reconstruction(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 2:
            trainer = Trainer_HDR(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 3:
            trainer = Trainer_CloudInpaiting(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 4:
            trainer = Trainer_MotionFusion(device, cfgs)
            trainer.run()
    elif cfgs.trainset.name == 'dsec':
        if cfgs.trainset.stage == 1:
            trainer = Trainer_reconstruction(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 2:
            trainer = Trainer_HDR(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 3:
            trainer = Trainer_CloudInpaiting(device, cfgs)
            trainer.run()
        elif cfgs.trainset.stage == 4:
            trainer = Trainer_MotionFusion(device, cfgs)
            trainer.run()

@hydra.main(config_path='conf', config_name='trainer')
def main(cfgs: DictConfig):
    # set num_workers of data loader
    if not cfgs.debug:
        n_devices = max(torch.cuda.device_count(), 1)
        cfgs.trainset.n_workers = min(os.cpu_count(), cfgs.trainset.n_workers * n_devices)
        cfgs.valset.n_workers = min(os.cpu_count(), cfgs.valset.n_workers * n_devices)
    else:
        cfgs.trainset.n_workers = 0
        cfgs.valset.n_workers = 0

    # resolve configurations
    if cfgs.ckpt.path is not None and cfgs.ckpt.resume:
        assert os.path.isfile(os.path.join(hydra.utils.get_original_cwd(), cfgs.ckpt.path))
        assert os.path.dirname(os.path.join(hydra.utils.get_original_cwd(), cfgs.ckpt.path))[-5:] == 'ckpts'
        shutil.rmtree(os.getcwd(), ignore_errors=True)
        cfgs.log.dir = os.path.dirname(os.path.dirname(cfgs.ckpt.path))

    if cfgs.log.dir is None:
        shutil.rmtree(os.getcwd(), ignore_errors=True)

        # run_name = ''
        # if cfgs.log.ask_name:
        #     run_name = input('Name your run (leave blank for default): ')
        # if run_name == '':
        run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

        cfgs.log.dir = os.path.join('outputs', cfgs.model.name, run_name)

        os.makedirs(os.path.join(hydra.utils.get_original_cwd(), cfgs.log.dir), exist_ok=False)

    if cfgs.port == 'random':
        cfgs.port = random.randint(10000, 20000)

    if 'override' in cfgs:
        cfgs = override_cfgs(cfgs, cfgs.override)

    if cfgs.training.accum_iter > 1:
        cfgs.model.batch_size //= int(cfgs.training.accum_iter)

    # create trainers
    os.chdir(hydra.utils.get_original_cwd())
    print(torch.cuda.device_count())
    if torch.cuda.device_count() == 0:  # CPU
        create_trainer(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_trainer(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_trainer, (cfgs,), torch.cuda.device_count())


if __name__ == '__main__':
    main()
