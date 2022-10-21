#!/usr/bin/python
# -*- coding:utf8 -*-


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from collections import OrderedDict

from .ConvVideoTransformer import ConvTransformer
from ..base import BaseModel

from thirdparty.deform_conv import DeformConv, ModulatedDeformConv
from posetimation.layers import BasicBlock, ChainOfBasicBlocks, DeformableCONV, PAM_Module, CAM_Module
from posetimation.layers import RSB_BLOCK, CHAIN_RSB_BLOCKS
from ..backbones.hrnet import HRNet
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DcPose_RSN(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(DcPose_RSN, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.inplanes = 64
        self.use_warping_train = cfg['MODEL']['USE_WARPING_TRAIN']
        self.use_warping_test = cfg['MODEL']['USE_WARPING_TEST']
        self.freeze_weights = cfg['MODEL']['FREEZE_WEIGHTS']
        self.use_gt_input_train = cfg['MODEL']['USE_GT_INPUT_TRAIN']
        self.use_gt_input_test = cfg['MODEL']['USE_GT_INPUT_TEST']
        self.warping_reverse = cfg['MODEL']['WARPING_REVERSE']
        self.cycle_consistency_finetune = cfg['MODEL']['CYCLE_CONSISTENCY_FINETUNE']

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.is_train = True if phase == TRAIN_PHASE else False
        # define rough_pose_estimation
        self.use_prf = cfg.MODEL.USE_PRF
        self.use_ptm = cfg.MODEL.USE_PTM
        self.use_pcn = cfg.MODEL.USE_PCN

        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        #self.freeze_hrnet_weights = False
        
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.use_rectifier = cfg.MODEL.USE_RECTIFIER
        self.use_margin = cfg.MODEL.USE_MARGIN
        self.use_group = cfg.MODEL.USE_GROUP

        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE
        ####
        self.rough_pose_estimation_net = HRNet(cfg, phase)

        self.pretrained = cfg.MODEL.PRETRAINED

        k = 3

        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM

        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM

        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        hyper_parameters = OrderedDict({
            "k": k,
            "prf_basicblock_num": prf_basicblock_num,
            "prf_inner_ch": prf_inner_ch,
            "ptm_basicblock_num": ptm_basicblock_num,
            "ptm_inner_ch": ptm_inner_ch,
            "prf_ptm_combine_basicblock_num": prf_ptm_combine_basicblock_num,
            "prf_ptm_combine_inner_ch": prf_ptm_combine_inner_ch,
        }
        )
        self.logger.info("###### MODEL {} Hyper Parameters ##########".format(self.__class__.__name__))
        self.logger.info(hyper_parameters)

        assert self.use_prf and self.use_ptm and self.use_pcn and self.use_margin and self.use_margin and self.use_group
        
    def _offset_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def _mask_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv


    # 220911 - multi head 형태로 변경 
    # 3*3 conv stack 이후 2개의 multi head로 변경해서 작업
    # gt와 motion gt의 차이를 구하는 형태로 구성 
    
    # def forward(self, x, margin, debug=False, vis=False):
    def forward(self, x, **kwargs):
        num_color_channels = 3
        assert "margin" in kwargs
        margin = kwargs["margin"]
        if not x.is_cuda or not margin.is_cuda:
            x.cuda()
            margin.cuda()

        if not self.use_rectifier:
            target_image = x[:, 0:num_color_channels, :, :]
            rough_x = self.rough_pose_estimation_net(target_image)
            return rough_x

        # current / previous / next
        # rough_pose_estimation is hrnet 
        
        # 왜 batchsize 방향으로 처음부터 작업을 하는게 아니라? 
        # 네트워크해서 채널방향으로 쪼개고 나서 그것을 batchsize방향으로 합치는 것일까?
        rough_heatmaps = self.rough_pose_estimation_net(torch.cat(x.split(num_color_channels, dim=1), 0))
        
        hrnet_stage3_output = rough_heatmaps[1]
        rough_heatmaps = rough_heatmaps[0]
        
        #print(rough_heatmaps.shape)
        #print(hrnet_stage3_output.shape)
        
        true_batch_size = int(rough_heatmaps.shape[0] / 3)

        # rough heatmaps in sequence frames
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps = rough_heatmaps.split(true_batch_size, dim=0)


        return  current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        rough_pose_estimation_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                rough_pose_estimation_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, DeformConv):
                filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                                     dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            elif isinstance(module, ModulatedDeformConv):
                filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                                     dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        ## ===============================================================
        ## 앞 과정은 정확하게 말하면 weight값이 없을 때, 초기화가 되는 과정이다. 
        ## 아래 과정을 통해서.... 

        # name = layer name !! 
        # _ = weight value !!  -> param.shape ,  param.requires_grad
        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                
                # self.pretrained_layers => * 이 정의되어 있음.!!
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        # 이 과정이 있는 이유는 ... coco로 학습된 pretrained 모델을 불러오기 때문에 
                        # 해당 과정에서 layer이름에 rough_pose_estimation_net가 앞에 더 붙게 된다. 
                        # 즉 이 부분을 보완하기 위해 작업이 된 것이라고 보면 좋다. 
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
                            
                # if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] is '*':
                if name in parameters_names or name in buffers_names:
                        # logger.info('=> init {} from {}'.format(name, pretrained))
                        print('=> init {}'.format(name))
                        need_init_state_dict[name] = m  
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))

        # rough_pose_estimation
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM
        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM
        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        if "DILATION" in cfg.MODEL.DEFORMABLE_CONV:
            dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
            dilation_str = ",".join(map(str, dilation))
        else:
            dilation_str = ""
        hyper_parameters_setting = "chPRF_{}_nPRF_{}_chPTM_{}_nPTM_{}_chComb_{}_nComb_{}_D_{}".format(
            prf_inner_ch, prf_basicblock_num, ptm_inner_ch, ptm_basicblock_num, prf_ptm_combine_inner_ch, prf_ptm_combine_basicblock_num,
            dilation_str)


        return hyper_parameters_setting

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = DcPose_RSN(cfg, phase, **kwargs)
        return model
