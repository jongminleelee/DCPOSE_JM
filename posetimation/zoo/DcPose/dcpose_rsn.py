#!/usr/bin/python
# -*- coding:utf8 -*-


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from collections import OrderedDict

from ..base import BaseModel

from thirdparty.deform_conv import DeformConv, ModulatedDeformConv
from posetimation.layers import BasicBlock, ChainOfBasicBlocks, DeformableCONV, PAM_Module, CAM_Module
from posetimation.layers import RSB_BLOCK, CHAIN_RSB_BLOCKS
from ..backbones.hrnet import HRNet
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY


class FlowLayer(nn.Module):

    def __init__(self, channels=17, bath_size=64, n_iter=10):
        super(FlowLayer, self).__init__()
        #self.bottleneck = nn.Conv3d(channels, bottleneck, stride=1, padding=0, bias=False, kernel_size=1)
        #self.unbottleneck = nn.Conv3d(bottleneck*2, channels, stride=1, padding=0, bias=False, kernel_size=1)
        #self.bn = nn.BatchNorm3d(channels)
        #channels = bottleneck
        
        
        # transpose() 는 딱 2개의 차원을 맞교환할 수 있다. 
        # permute() 는 모든 차원들을 맞교환할 수 있다.
        # x = torch.rand(16, 32, 3)
        # y = x.tranpose(0, 2)  # [3, 32, 16]
        # z = x.permute(2, 1, 0)  # [3, 32, 16]
        
        # (batch, channel, height, width)
        
        self.n_iter = n_iter
            
        # h 방향, w 방향으로 각각의 gradient를 계산하기 위해서로 보인다. 
        # params의 차이는 - require_grad 를 반영할건지, 반영 안할 건지에 대한 부분이다. 
        
        # torch.Size([1, 1, 1, 3])
        # conv 연산을 통해서 gradient를 구하는 부분
        self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,channels,1,1))
        # torch.Size([1, 1, 3, 1]) -> transpoe(3,2)를 통해 변경 
        # conv 연산을 통해서 gradient를 구하는 부분
        self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,channels,1,1))
            

        self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))
        self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))
        self.div = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))
        self.div2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))            

        self.channels = channels
        
        self.t = 0.3
        self.l = 0.15
        self.a = 0.25        

        self.t = nn.Parameter(torch.FloatTensor([self.t]))
        self.l = nn.Parameter(torch.FloatTensor([self.l]))
        self.a = nn.Parameter(torch.FloatTensor([self.a]))

        self.conv = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)
        
        # 수정
        self.bn = nn.BatchNorm2d(channels*2)
        self.relu = nn.ReLU(inplace=True)


    def norm_img(self, x):
        mx = torch.max(x)
        mn = torch.min(x)
        x = 255*(x-mn)/(mn-mx)
        return x
            
            
    # (batch, channel, height, width)        
    def forward_grad(self, x):
        grad_x = F.conv2d(F.pad(x, (0,0,0,1)), self.f_grad)#, groups=self.channels)
        grad_x[:,:,-1,:] = 0
        
        grad_y = F.conv2d(F.pad(x, (0,0,0,1)), self.f_grad2)#, groups=self.channels)
        grad_y[:,:,-1,:] = 0
        return grad_x, grad_y


    # divergence : 발산 
    # 왜 있는가? 
    # (batch, channel, height, width)
    # torch.nn.functional.pad(input, pad, mode='constant', value=0) => 원하는대로 padding을 줄 수 있는 함수!!.
    def divergence(self, x, y):
        
        # -1 부분 있으면 마지막 부분을 하나를 제외한다는 의미이다. 
        # to pad the last 2 dimensions of the input tensor, then use (left, right, top, bottom);
        # 첫 행 부분에 패딩을 추가하고, 마지막 부분의 값은 제외한다는 의미이다.
        tx = F.pad(x[:,:,:-1,:], (0,0,1,0))
        ty = F.pad(y[:,:,:-1,:], (0,0,1,0))
        
        # 해당 부분의 코드의미도 tx ty 관련해서 행 마지막 부분에 패딩을 추가한다는 의미이다. 
        # 해당 부분을 통해서 gradient를 각각 계산하게 되는데 ................
        # 왜 구지 처음 x,y 값에 대한 첫줄을 패딩으로 추가해주는 것일까? 그냥 이미지에서 마지막 bottom 부분에만 패딩을 추가해주면 될 것 같은데 ...... 
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # self.div = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,channels,1,1))
        grad_x = F.conv2d(F.pad(tx, (0,0,0,1)), self.div)#, groups=self.channels)
        grad_y = F.conv2d(F.pad(ty, (0,0,0,1)), self.div2)#, groups=self.channels)
        return grad_x + grad_y
        
        
    def forward(self, current, next):
        # flowlayer를 통과하게 되면 n프레임을 입력하게 되면 n-1개의 feature vector가 나오게 됨. 
        # 즉 ... 최정적으로 feature vector를 구할 때 기존 프레임의 값을 합치는 것이 성능이 잘 나오기 때문에 ... residual 부분에서 마지막 프레임을 제외하고 더해주는 것이다.!! 
        #residual = x[:,:,:-1]
        
        # 
        #x = self.bottleneck(x)
        
        # b,c,t,h,w
        
        # x라는 것은 전체 비디오 n프레임이 있을 경우 => 1 ~ n-1 프레임까지를 말한다. 
        #x = inp[:,:,:-1]
        
        # y라는 것은 전체 비디오 n프레임 중에서 => 2 ~ n프레임까지를 말한다. 
        # y라는 값은 x 기준으로 보았을 때 한 프레임정도 앞에 있는 것들 말한다. 
        #y = inp[:,:,1:]
        
        
        # => x,y 를 각각 flowlayer의 전프레임과 후프레임으로 입력하여 각 레이어에 대한 flow 값을 도출한다. 
        
        #b,c,t,h,w = x.size()
        #x = x.permute(0,2,1,3,4).contiguous().view(b*t,c,h,w)
        #y = y.permute(0,2,1,3,4).contiguous().view(b*t,c,h,w)
        
        x = self.norm_img(current)
        y = self.norm_img(next)
        
        u1 = torch.zeros_like(x)
        u2 = torch.zeros_like(y)
        l_t = self.l * self.t
        taut = self.a/self.t

        # (batch, channel, height, width)
        # padding : (left, right, top, bottom);
        # 왜 여기서는 left와 right에 대한 패딩을 한 이유는 gradient를 구할 경우, 3개의 픽셀씩 묶어서 작업을 하기 때문에 
        # 패딩을 하지 않을 경우, w 너비 부분의 크기가 줄어둘기 때문이다. 
        grad2_x = F.conv2d(F.pad(y,(1,1,0,0)), self.img_grad, padding=0, stride=1)#, groups=self.channels)
        grad2_x[:,:,:,0] = 0.5 * (x[:,:,:,1] - x[:,:,:,0])
        grad2_x[:,:,:,-1] = 0.5 * (x[:,:,:,-1] - x[:,:,:,-2])

        # (batch, channel, height, width)
        # padding : (left, right, top, bottom);
        grad2_y = F.conv2d(F.pad(y, (0,0,1,1)), self.img_grad2, padding=0, stride=1)#, groups=self.channels)
        grad2_y[:,:,0,:] = 0.5 * (x[:,:,1,:] - x[:,:,0,:])
        grad2_y[:,:,-1,:] = 0.5 * (x[:,:,-1,:] - x[:,:,-2,:])
        
        p11 = torch.zeros_like(x.data)
        p12 = torch.zeros_like(x.data)
        p21 = torch.zeros_like(x.data)
        p22 = torch.zeros_like(x.data)

        gsqx = grad2_x**2
        gsqy = grad2_y**2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        
        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12


            # 왜? v1, v2 두개의 값이 있는가??
            # v1과 v2는 각각 x와 y의 방향에 대한 크기이다.
            v1 = torch.zeros_like(x.data)
            v2 = torch.zeros_like(x.data)
            mask1 = (rho < -l_t*grad).detach()
            v1[mask1] = (l_t * grad2_x)[mask1]
            v2[mask1] = (l_t * grad2_y)[mask1]
            
            # 해당 mask2는아래 연산을 통해서 각 인덱싱부분에 값을 넣을지 안 넣을지를 결정해주는 부분이다. 
            # 문법적 이해 체크하도록!!
            mask2 = (rho > l_t*grad).detach()
            v1[mask2] = (-l_t * grad2_x)[mask2]
            v2[mask2] = (-l_t * grad2_y)[mask2]

            # ^1 ^2 의 의미는 무엇인가?
            mask3 = ((mask1==False) & (mask2==False) & (grad > 1e-12)).detach()
            v1[mask3] = ((-rho/grad) * grad2_x)[mask3]
            v2[mask3] = ((-rho/grad) * grad2_y)[mask3]
            
            # delete a Tensor in GPU 
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            # u1은 x방향 u2는 y방향을 의미한다. 
            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)
            
            # delete a Tensor in GPU 
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)
            
            p11 = (p11 + taut * u1x) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            
            # delete a Tensor in GPU 
            del u1x
            del u1y
            del u2x
            del u2y
         
        
        flow = torch.cat([u1,u2], dim=1)
        #flow = self.conv(flow)
        flow = self.bn(flow)
        #flow = self.relu(flow)
        
        
        #print(u1)
        #print(u2)
        
        return flow
        # 조인트별로 x방향과 y방향 motion estimator이다. 
        # bath, channel 


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

        ####### PRF #######
        #diff_temporal_fuse_input_channels = self.num_joints * 4
        #self.diff_temporal_fuse = CHAIN_RSB_BLOCKS(diff_temporal_fuse_input_channels, prf_inner_ch, prf_basicblock_num,
        #                                           )

        # self.diff_temporal_fuse = ChainOfBasicBlocks(diff_temporal_fuse_input_channels, prf_inner_ch, 1, 1, 2,
        #                                              prf_basicblock_num, groups=self.num_joints)

        ####### PTM #######
        if ptm_basicblock_num > 0:

            self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 3, ptm_inner_ch, ptm_basicblock_num,
                                                          )

            # self.support_temporal_fuse = ChainOfBasicBlocks(self.num_joints * 3, ptm_inner_ch, 1, 1, 2,
            #                                                 ptm_basicblock_num, groups=self.num_joints)
        else:
            self.support_temporal_fuse = nn.Conv2d(self.num_joints * 3, ptm_inner_ch, kernel_size=3, padding=1,
                                                   groups=self.num_joints)

        #prf_ptm_combine_ch = prf_inner_ch + ptm_inner_ch
        #prf_ptm_combine_ch = prf_inner_ch + ptm_inner_ch + 34
        
        #prf_ptm_combine_ch = ptm_inner_ch + 17
        prf_ptm_combine_ch = ptm_inner_ch

        self.offset_mask_combine_conv = CHAIN_RSB_BLOCKS(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, prf_ptm_combine_basicblock_num)
        # self.offset_mask_combine_conv = ChainOfBasicBlocks(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, 1, 1, 2,
        #                                                    prf_ptm_combine_basicblock_num)


        self.p_c_heatmap_output_layer = CHAIN_RSB_BLOCKS(161, 17, 3)
        self.n_c_heatmap_output_layer = CHAIN_RSB_BLOCKS(161, 17, 3)


        ###### motion_module #######
        self.motion_layer1 = FlowLayer(48,8)
        
        self.motion_layer2 = FlowLayer(48,8)



        ####### PCN #######
        self.offsets_list, self.masks_list, self.modulated_deform_conv_list = [], [], []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            # offsets
            offset_layers, mask_layers = [], []
            offset_layers.append(self._offset_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            mask_layers.append(self._mask_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            self.offsets_list.append(nn.Sequential(*offset_layers))
            self.masks_list.append(nn.Sequential(*mask_layers))
            self.modulated_deform_conv_list.append(DeformableCONV(self.num_joints, k, dilation))

        self.offsets_list = nn.ModuleList(self.offsets_list)
        self.masks_list = nn.ModuleList(self.masks_list)
        self.modulated_deform_conv_list = nn.ModuleList(self.modulated_deform_conv_list)

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


        # ===================================================================================

        #true_batch_size2 = int(rough_heatmaps.shape[0] / 3)

        # hrnet stage3 출력값을 활용하여 flowlayer 작업을 진행을 함.
        # 기본적으로 stage3_output => batchsize, 48, 96, 72  형태를 가지고 있음.
        current_hrnet_stage3_output, previous_hrnet_stage3_output, next_hrnet_stage3_output = hrnet_stage3_output.split(true_batch_size, dim=0)


        # motion_module_flowlayer
        # 48채널 * 2 형태로 출력이 됨.
        flow_p_c = self.motion_layer1(previous_hrnet_stage3_output,current_hrnet_stage3_output)
        flow_n_c = self.motion_layer2(next_hrnet_stage3_output,current_hrnet_stage3_output)
        
        # 48채
        stage3_p_c_diff = current_hrnet_stage3_output - previous_hrnet_stage3_output
        stage3_n_c_diff = current_hrnet_stage3_output - next_hrnet_stage3_output

        p_c_relation_output = torch.cat([previous_rough_heatmaps,flow_p_c,stage3_p_c_diff], dim=1)
        n_c_relation_output = torch.cat([next_rough_heatmaps,flow_n_c,stage3_n_c_diff], dim=1)
        
        p_c_heatmap_output = self.p_c_heatmap_output_layer(p_c_relation_output)
        n_c_heatmap_output = self.n_c_heatmap_output_layer(n_c_relation_output)
        
        #print(p_c_heatmap_output.shape)
        #print(n_c_heatmap_output.shape)

        # jongmin 코드 기반으로 작업된 부분이다. 
        support_heatmaps = torch.cat([current_rough_heatmaps,p_c_heatmap_output*0.5,n_c_heatmap_output*0.5], dim=1)
        support_heatmaps = self.support_temporal_fuse(support_heatmaps).cuda()       
        
          
        '''          
        # Difference A and Difference B
        diff_A = current_rough_heatmaps - previous_rough_heatmaps
        diff_B = current_rough_heatmaps - next_rough_heatmaps

        # default use_margin,use_group
        # 왼쪽마진와 오른족 마진에 대한 값이 때문에 두개의 값을 합하게 되면 n-p와 같은 효과를 볼 수 있다. 
        interval = torch.sum(margin, dim=1, keepdim=True)  # interval = n-p
        
        # pre_weight와 next_weight 계산 시, 브로드캐스팅을 통해서 연산이 된다. 
        # 왜?? unsqueeze를 3번이나 하는 것인가?
        margin = torch.div(margin.float(), interval.float())  # margin -> (c-p)/(n-p), (n-c)/(n-p)
        prev_weight, next_weight = margin[:, 1], margin[:, 0]  # previous frame weight - (n-c)/(n-p) , next frame weight - (c-p)/(n-p)
        diff_shape = diff_A.shape
        prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        extend_shape = [1, 1]  # batch, channel
        extend_shape.extend(list(diff_shape[2:]))
        prev_weight, next_weight = prev_weight.repeat(extend_shape), next_weight.repeat(extend_shape)

        #조인트별로 쪼개는 부분이다. 
        #조인트별로 쪼개서 비교한다라고 생각하면 된다. 
        #diff_A_list, diff_B_list = diff_A.split(1, dim=1), diff_B.split(1, dim=1)
        #temp_diff_fuse_list = []
        #for joint_index in range(self.num_joints):
        #    temp_diff_fuse_list.append(diff_A_list[joint_index])
        #    temp_diff_fuse_list.append(diff_A_list[joint_index] * prev_weight)
        #    temp_diff_fuse_list.append(diff_B_list[joint_index])
        #    temp_diff_fuse_list.append(diff_B_list[joint_index] * next_weight)

        #dif_heatmaps = torch.cat(temp_diff_fuse_list, dim=1)
        #dif_heatmaps = self.diff_temporal_fuse(dif_heatmaps)

        # 조인트별 기준으로 쪼개지게 된다. 
        # 히트맵의 결과값은 batch, joint_num 형태로 구성되기 때문이다.
        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)
        temp_support_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_support_fuse_list.append(current_rough_heatmaps_list[joint_index])
            temp_support_fuse_list.append(previous_rough_heatmaps_list[joint_index] * prev_weight)
            temp_support_fuse_list.append(next_rough_heatmaps_list[joint_index] * next_weight)

        support_heatmaps = torch.cat(temp_support_fuse_list, dim=1)
        
        # self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 3, ptm_inner_ch, ptm_basicblock_num,)
        # 해당 위 layer는 3*3 stack layer 부분이다. 
        # 이 때 왜? ptm의 결과를 
        support_heatmaps = self.support_temporal_fuse(support_heatmaps).cuda()
        '''
        # 3*3 conv stack conv 처리 !!
        prf_ptm_combine_featuremaps = self.offset_mask_combine_conv(torch.cat([support_heatmaps], dim=1))
        #prf_ptm_combine_featuremaps = self.offset_mask_combine_conv(torch.cat([dif_heatmaps, support_heatmaps], dim=1))
        
        # jongmin - add code
        # prf_ptm_combine_featuremaps2 = self.offset_mask_combine_conv(torch.cat([dif_heatmaps, support_heatmaps], dim=1))


          #DEFORMABLE_CONV:
            #DILATION:
            #- 3
            #- 6
            #- 9
            #- 12
            #- 15

        warped_heatmaps_list = []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            offsets = self.offsets_list[d_index](prf_ptm_combine_featuremaps)
            masks = self.masks_list[d_index](prf_ptm_combine_featuremaps)

            warped_heatmaps = self.modulated_deform_conv_list[d_index](support_heatmaps, offsets, masks)
            warped_heatmaps_list.append(warped_heatmaps)


        if self.deformable_aggregation_type == "weighted_sum":

            # 5개의 dilations가 있기 때문에 해당 부분을 균등하게 1/5 씩 weight값을 부여함. 
            warper_weight = 1 / len(self.deformable_conv_dilations)
            output_heatmaps = warper_weight * warped_heatmaps_list[0]
            for warper_heatmaps in warped_heatmaps_list[1:]:
                output_heatmaps += warper_weight * warper_heatmaps

        else:
            output_heatmaps = self.deformable_aggregation_conv(torch.cat(warped_heatmaps_list, dim=1))
            # elif self.deformable_aggregation_type == "conv":

        # ----------------------------------------------------------------------

        # warped_heatmaps_list2 = []
        # for d_index, dilation in enumerate(self.deformable_conv_dilations):
        #     offsets2 = self.offsets_list[d_index](prf_ptm_combine_featuremaps)
        #     masks2 = self.masks_list[d_index](prf_ptm_combine_featuremaps)

        #     warped_heatmaps2 = self.modulated_deform_conv_list[d_index](support_heatmaps, offsets2, masks2)
        #     warped_heatmaps_list2.append(warped_heatmaps2)


        # if self.deformable_aggregation_type == "weighted_sum":

        #     # 5개의 dilations가 있기 때문에 해당 부분을 균등하게 1/5 씩 weight값을 부여함. 
        #     warper_weight2 = 1 / len(self.deformable_conv_dilations)
        #     output_heatmaps2 = warper_weight2 * warped_heatmaps_list2[0]
        #     for warper_heatmaps2 in warped_heatmaps_list2[1:]:
        #         output_heatmaps2 += warper_weight2 * warper_heatmaps2


        # self.freeze_hrnet_weights => true (config파일)
        
        # jongmin add code    
        # output_heatmaps : motion gt와 비교
        # output_heatmaps2 : origin gt와 비교
        
        # p->c, n->c 관련된 output도 추가한다. 각각 gt와 비교해서 loss를 구한다.
        return output_heatmaps, p_c_heatmap_output, n_c_heatmap_output

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

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
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
