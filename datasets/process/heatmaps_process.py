#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import cv2

from .affine_transform import get_affine_transform, exec_affine_transform


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):  # heatmap [batch,channel,width,height]
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):  # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = exec_affine_transform(coords[p, 0:2], trans)
    return target_coords


# 220911 : song generate_heatmap 생성코드 
# 
def generate_heatmaps(joints, joints_vis, sigma, image_size, heatmap_size, num_joints, optical_image, **kwargs):
    """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param sigma:
        :param image_size:
        :param heatmap_size:
        :param num_joints:
        :return: target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        
        image_x = int(joints[joint_id][0])
        image_y = int(joints[joint_id][1])
        
        
        #print("mu_x : ")
        #print(mu_x)
        #print("image_size : {}".format(image_size))
        
        # opticlflow 이미지 기반으로 하여, 모션 heatmap 생성에 대한 부분 필요. 
        # heatmap generator 시, opticalimage를 받아서 hsv로 변환한 다음. 
        # angle 값과 vector 크기값을 호출해야함. 
        
        # 확대 축소 flip 등에 대한 과정이 진행되면서 조인트의 값이 보이기도 하고, 안 보이는 상황이 발생하게 된다. 
        # 즉 안보이는 것에 대해서는 target_weight[:, 0] = joints_vis[:, 0] 의 값을 통해서 해결을 해야하는 상황이다. !! 
        g = np.zeros((19, 19))
        if optical_image is None:
            optical_direction = 0
            pass
        
        # mu_x 와 mu_y 는 수정된 히트맵 크기의 x,y좌표이기 때문에 ... 
        # opticalflow 이미지 방향과는 맞지 않는다.     
            
        elif image_x < 1 or image_x >= 287 or image_y < 1 or image_y >= 383:
            optical_direction = 0
            pass        
        else:
            # HSV format 변환 ----- 22.08.24 inpyo
            hsv_image = cv2.cvtColor(optical_image, cv2.COLOR_RGB2HSV)
            hsv_image = np.array(hsv_image, dtype='float32')
            hsv_image[..., 0] = hsv_image[..., 0] * 2
            # hsv_image[..., 1] = hsv_image[..., 1] * 100
            # hsv_image[..., 2] = hsv_image[..., 2] * 100

        # area optical flow (3x3) // 22.08.27 inpyo
        #     optical_direction = hsv_image[image_y, image_x, 0]
        #     optical_mag = hsv_image[image_y, image_x, 1]
            
            optical_direction = (hsv_image[image_y, image_x, 0] + 0.94595945 * (hsv_image[image_y-1, image_x, 0]
                                 + hsv_image[image_y, image_x-1, 0] + hsv_image[image_y+1, image_x, 0] + hsv_image[image_y, image_x+1, 0])
                                 + 0.8948393 * (hsv_image[image_y-1, image_x-1, 0] + hsv_image[image_y+1, image_x+1, 0]
                                 + hsv_image[image_y+1, image_x-1, 0] + hsv_image[image_y-1, image_x+1, 0]))/9
            optical_direction = optical_direction.astype(np.uint8)

            optical_mag = (hsv_image[image_y, image_x, 1] + 0.94595945 * (hsv_image[image_y-1, image_x, 1]
                                 + hsv_image[image_y, image_x-1, 1] + hsv_image[image_y+1, image_x, 1] + hsv_image[image_y, image_x+1, 1])
                                 + 0.8948393 * (hsv_image[image_y-1, image_x-1, 1] + hsv_image[image_y+1, image_x+1, 1]
                                 + hsv_image[image_y+1, image_x-1, 1] + hsv_image[image_y-1, image_x+1, 1]))/9
            optical_mag = optical_mag.astype(np.uint8)

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2

            if (optical_image is None) or (optical_mag < 50):
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            else:
                g1 = np.exp(- ((x - x0 + 1 * math.cos(math.pi * optical_direction / 180)) ** 2 + (
                        y - y0 - 1 * math.sin(math.pi * optical_direction / 180)) ** 2) / (2 * sigma ** 2)) * 3 / 5

                size = 21
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                g2 = np.exp(- ((x - x0 - (2+optical_mag/300)*math.cos(math.pi*optical_direction/180)) ** 2 + (y - y0 + (2+optical_mag/300)*math.sin(math.pi*optical_direction/180)) ** 2) / (2 * sigma ** 2))/10

                size = 23
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                g3 = np.exp(- ((x - x0 - (4+2*optical_mag/300)*math.cos(math.pi*optical_direction/180)) ** 2 + (y - y0 + (4+2*optical_mag/300)*math.sin(math.pi*optical_direction/180)) ** 2) / (2 * sigma ** 2))*3/10

                g = g1 + g2[:19, :19] + g3[:19, :19]
            # elif optical_mag >= 200:
            #     g1 = np.exp(- ((x - x0 + 1 * math.sin(math.pi * optical_direction / 180)) ** 2 + (
            #             y - y0 - 1 * math.cos(math.pi * optical_direction / 180)) ** 2) / (2 * sigma ** 2)) * 3 / 5
            #
            #     size = 21
            #     x = np.arange(0, size, 1, np.float32)
            #     y = x[:, np.newaxis]
            #     g2 = np.exp(- ((x - x0 - 2 * math.sin(math.pi * optical_direction / 180)) ** 2 + (
            #             y - y0 + 2 * math.cos(math.pi * optical_direction / 180)) ** 2) / (2 * sigma ** 2)) / 10
            #
            #     size = 23
            #     x = np.arange(0, size, 1, np.float32)
            #     y = x[:, np.newaxis]
            #     g3 = np.exp(- ((x - x0 - 6 * math.sin(math.pi * optical_direction / 180)) ** 2 + (
            #             y - y0 + 6 * math.cos(math.pi * optical_direction / 180)) ** 2) / (2 * sigma ** 2)) * 3 / 10
            #     g = g1 + g2[:19, :19] + g3[:19, :19]
        
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if ("use_different_joints_weight" in kwargs) and (kwargs["use_different_joints_weight"]):
        target_weight = np.multiply(target_weight, kwargs["joints_weight"])

    return target, target_weight


#220911 : origin gt 생성을 위한 motion gt와 구분해서 처리.
def generate_heatmaps_origin(joints, joints_vis, sigma, image_size, heatmap_size, num_joints, **kwargs):
    """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param sigma:
        :param image_size:
        :param heatmap_size:
        :param num_joints:
        :return: target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if ("use_different_joints_weight" in kwargs) and (kwargs["use_different_joints_weight"]):
        target_weight = np.multiply(target_weight, kwargs["joints_weight"])

    return target, target_weight