import torch
import os 
from collections import OrderedDict
from models.hdr_model.base_model import BaseModel
from models.hdr_model import networks
from models.hdr_model.vgg import Vgg16

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                
    return (batch - mean) / std


def Gram_matrix(input):
    a,b,c,d = input.size()
    features = input.reshape(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)


def load_module_dict(pth_path, gpu_ids):
    kwargs={'map_location': lambda storage, loc: storage.cuda(gpu_ids)}
    state_dict = torch.load(pth_path)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    # load params
    return new_state_dict


class VideoModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance')

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.netUpsample = networks.define_UpsampleNet(gpu_ids=self.gpu_ids, scale=opt.up_scale)
        self.netLumiFusion = networks.define_G(init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netColor = networks.define_ColorNet(netColor=opt.netColor, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, n_blocks=opt.colornet_n_blocks, state_nc=opt.state_nc, gpu_ids=self.gpu_ids)
        
        # for infer
        if opt.phase == "infer":
            self.visual_names = ['input_ldr_rgb', 'input_im', 'output_hdr_rgb']
            self.model_names = ['LumiFusion', 'Color']
            self.netLumiFusion = self.__load_networks(netType='LumiFusion')
            self.netColor = self.__load_networks(netType='Color')
        
        # initialize hidden state to None
        self.last_state = None

    
    def forward(self, input):
        self.input_im_up = self.netUpsample(input['intensity_image'])
        self.output_hdr_y, self.att_map = self.netLumiFusion(input['ldr_y'], self.input_im_up.detach())
        self.output_hdr_rgb, self.last_state = self.netColor(self.output_hdr_y.detach(), input['ldr_u'], input['ldr_v'], self.last_state)


    def __load_networks(self, netType):
        if netType == 'LumiFusion':
            net = self.netLumiFusion
            load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'luminance_fusion_net.pth')
        elif netType == 'Color':
            net = self.netColor
            load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'chrominance_compensation_net.pth')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)
        net.eval()
        return net
    
    
    def optimize_parameters(self):
        pass