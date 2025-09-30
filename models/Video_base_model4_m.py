import logging
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2, VGGLoss,PerceptualLoss

import time

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)
            self.cri_vgg = VGGLoss()
            self.cri_perceptual = PerceptualLoss(OrderedDict([('conv1_2', 1), ('conv2_2', 1), ('conv3_4', 1), ('conv4_4', 1)]))
            # self.L_color = L_color_zy()

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        snr = torch.div(light, noise + 0.0001)

        batch_size = snr.shape[0]
        height = snr.shape[2]
        width = snr.shape[3]
        snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0]
        snr_max = snr_max.view(batch_size, 1, 1, 1)
        snr_max = snr_max.repeat(1, 1, height, width)
        snr = snr * 1.0 / (snr_max+0.0001)

        snr = torch.clamp(snr, min=0, max=1.0)
        snr = snr.float()


        self.fake_H ,self.out_fea= self.netG(self.var_L, snr)

        h, w = self.out_fea.shape[2:]
        self.side_gt = torch.nn.functional.interpolate(self.real_H, (h, w), mode='bicubic', align_corners=False)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_side_pix1 = self.cri_pix(self.out_fea, self.side_gt)

        l_percep, _ = self.cri_perceptual(self.fake_H, self.real_H)
        l_side_percep, _ = self.cri_perceptual(self.out_fea, self.side_gt)
        # l_vgg = self.cri_vgg(self.fake_H, self.real_H) * 0.01
        # l_color = self.L_color(self.fake_H, self.real_H)+ l_color

        l_final = l_pix + l_side_pix1 * 0.8 + l_percep*0.01 + l_side_percep*0.8
        l_final.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            snr = torch.div(light, noise + 0.0001)

            batch_size = snr.shape[0]
            height = snr.shape[2]
            width = snr.shape[3]
            snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0]
            snr_max = snr_max.view(batch_size, 1, 1, 1)
            snr_max = snr_max.repeat(1, 1, height, width)
            snr = snr * 1.0 / (snr_max+0.0001)

            snr = torch.clamp(snr, min=0, max=1.0)
            snr = snr.float()
            self.fake_H = self.netG(self.var_L, snr)
        self.netG.train()

    def test4(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            snr = torch.div(light, noise + 0.0001)

            batch_size = snr.shape[0]
            height = snr.shape[2]
            width = snr.shape[3]
            snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0]
            snr_max = snr_max.view(batch_size, 1, 1, 1)
            snr_max = snr_max.repeat(1, 1, height, width)
            snr = snr * 1.0 / (snr_max + 0.0001)

            snr = torch.clamp(snr, min=0, max=1.0)
            snr = snr.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 400
            W_new = 608
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            snr = F.interpolate(snr, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, snr)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del snr
            torch.cuda.empty_cache()

        self.netG.train()


    def test5(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            snr = torch.div(light, noise + 0.0001)

            batch_size = snr.shape[0]
            height = snr.shape[2]
            width = snr.shape[3]
            snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0]
            snr_max = snr_max.view(batch_size, 1, 1, 1)
            snr_max =snr_max.repeat(1, 1, height, width)
            snr = snr * 1.0 / (snr_max + 0.0001)

            snr = torch.clamp(snr, min=0, max=1.0)
            snr = snr.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 384
            W_new = 384
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            snr = F.interpolate(snr, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, snr)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del snr
            torch.cuda.empty_cache()

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        if not(len(self.nf.shape) == 4):
            self.nf = self.nf.unsqueeze(dim=0)
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        snr = torch.div(light, noise + 0.0001)

        batch_size = snr.shape[0]
        height = snr.shape[2]
        width = snr.shape[3]
        snr_max = torch.max(snr.view(batch_size, -1), dim=1)[0]
        snr_max = snr_max.view(batch_size, 1, 1, 1)
        snr_max = snr_max.repeat(1, 1, height, width)
        snr = snr * 1.0 / (snr_max + 0.0001)

        snr = torch.clamp(snr, min=0, max=1.0)
        snr = snr.float()
        snr = snr.repeat(1, 3, 1, 1)
        out_dict['rlt3'] =snr[0].float().cpu()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H[0].detach()[0].float().cpu()
        out_dict['ill'] = snr[0].float().cpu()
        out_dict['rlt2'] = self.nf.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del dark
        del light
        del snr
        del noise
        del self.real_H
        del self.nf
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def add_Gaussian_noise_color(self,img, noise_level1=2, noise_level2=25, color_ratio=1):

        noise_level = random.randint(noise_level1, noise_level2)
        noise = torch.normal(0, noise_level / 255.0, img.shape, device='cuda')
        img_noisy = img + noise
        # img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
        # ֱ����PyTorch������ʹ��clamp����
        img_noisy = img_noisy.clamp(0.0, 1.0)
        return img_noisy

    def input_snr(self,image, prob_=0.75, value=0.1):
        """
        Multiplicative bernoulli using PyTorch
        """
        # print(image.shape)
        b, c, h, w = image.shape
        # x, y, z = image.shape[:3]  # ȥ�����һ��ά�ȣ�������������ά�Ȼ�������
        # ����һ����ͼ����ͬ��״�ĸ���������������ת��Ϊ��������
        snr = torch.bernoulli(torch.empty((b, 1, h, w), device='cuda').uniform_(0, prob_))
        # print(image.shape)
        snr = snr.repeat(1, c, 1, 1)
        noise_image = image * snr
        noise_image = noise_image - value + value * snr
        # print(noise_image.shape)

        return noise_image

    def input_snr_with_noise(self,img, sf=1, lq_patchsize=32, noise_level=15, if_snr=True, snr1=70, snr2=80):
        # print(img.shape)
        b, c, h, w = img.shape
        # img = img[:, :, :w - w % sf, :h - h % sf]  # mod crop
        # print(img.shape)
        b2, c2, h, w = img.shape

        if h < lq_patchsize * sf or w < lq_patchsize * sf:
            raise ValueError(f'img size ({h}X{w}) is too small!')

        hq = img

        if noise_level > 0:
            img = self.add_Gaussian_noise_color(img, noise_level1=noise_level, noise_level2=noise_level)
            # print(img.shape)
        # img, hq = random_crop(img, hq, sf, lq_patchsize)

        if if_snr:
            prob = random.randint(snr1, snr2) / 100
            # prob = 0.75
            img = self.input_snr(img, prob_=prob)

        return img
