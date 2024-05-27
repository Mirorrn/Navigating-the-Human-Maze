#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
from math import cos, sin, pi
import torch

class AugmentSelection:

    def __init__(self, flip=False, ):
        self.flip = flip  # shift y-axis

        self.img_h = 28
        self.img_w = 28

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) >= 0.5
        return AugmentSelection(flip)

    @staticmethod
    def unrandom():
        flip = False
        return AugmentSelection(flip)

    def affine(self):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards
        # look https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html

        #  width, height = img_shape
        degree = random.uniform(-1., 1.) * 15 if self.flip else 0.
        # degree = 7.

        A = cos(degree / 180. * pi)
        B = sin(degree / 180. * pi)

        rotate = np.array([[A, -B, 0],
                           [B, A, 0],
                           [0, 0, 1.]])

        center2zero = np.array([[1., 0., -(self.img_h / 2. -1)],
                                [0., 1., -(self.img_h / 2. -1)],
                                [0., 0., 1.]])

        flip = random.uniform(0., 1.) >= 0.5
        flip = -1. if flip else 1.
        flip_v = np.array([[flip, 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)

        flip = random.uniform(0., 1.) >= 0.5
        flip = -1. if flip else 1.
        flip_h = np.array([[1, 0., 0.],
                           [0., flip, 0.],
                           [0., 0., 1.]], dtype=np.float32)

        center2center = np.array([[1., 0., (self.img_w / 2. -1)],
                                 [0., 1.,  (self.img_w / 2. -1)],
                                 [0., 0., 1.]])

        # order of combination is reversed
        combined = center2center @ flip_v @ flip_h @ center2zero  # @ - matmul

        return combined[0:2]  # 3th row is not important anymore


# Augmentation like in Agentformer https://arxiv.org/abs/2103.14023
class Transformer:
    @staticmethod
    def print_traj(data_obs, data_pred_gt):  # only for debug
        data_obs = data_obs[0] #first batch to print
        data_pred_gt = data_pred_gt[0]
        fig, ax = plt.subplots()
        for x_data in data_obs:
            x = x_data[0, :]
            x = x[np.nonzero(x)]
            y = x_data[1, :]
            y = y[np.nonzero(y)]
            ax.plot(x, y, 'g', alpha=.3)
            ax.plot(x[:-1], y[:-1], 'g*', alpha=.5)
            ax.plot(x[-1], y[-1], 'gX', alpha=.5)

        for x_data in data_pred_gt:
            if x_data.any():
                x = x_data[0, :]
                x = x[np.nonzero(x)]
                y = x_data[1, :]
                y = y[np.nonzero(y)]
                ax.plot(x, y, 'g', alpha=.3)
                ax.plot(x[:-1], y[:-1], 'b*', alpha=.5)
                ax.plot(x[-1], y[-1], 'bX', alpha=.5)

        plt.show()
        plt.close

    @staticmethod
    def traj_matmul(original_points, M):
        for j , batch in enumerate(original_points):
            for i, o in enumerate(batch):
               # for k, o in enumerate(tr):
              #  o = o[np.nonzero(o)]
                mask = np.where(o.sum(axis=0) != 0, 1., 0.)
                ones = np.ones_like(o[0, :])
                ones = np.expand_dims(ones, axis=0)

                tmp= np.concatenate((o, ones),axis=0) # we reuse 3rd column in
                # completely different way here, it is hack for matmul with M
                original_points[j][i] = np.matmul(M, tmp)  * mask # transpose for multiplikation
        return original_points

    @staticmethod
    def rotation_2d_torch(x, theta, origin=None):
        if origin is None:
            origin = torch.zeros(2).to(x.device).to(x.dtype)
        norm_x = x - origin
        norm_rot_x = torch.zeros_like(x)
        norm_rot_x[:,:, 0] = norm_x[:,:, 0] * torch.cos(theta) - norm_x[:,:, 1] * torch.sin(theta)
        norm_rot_x[:,:, 1] = norm_x[:,:, 0] * torch.sin(theta) + norm_x[:,:, 1] * torch.cos(theta)
        rot_x = norm_rot_x + origin
        return rot_x, norm_rot_x

    @staticmethod
    def scene_origin(obs_traj_pos, pred_traj_gt_pos, seq_start_end):
        szene_origin = []
        for i, (start, end) in enumerate(seq_start_end):
            scene_size = end - start
            tmp_origin = torch.cat([obs_traj_pos[:, start:end],
                                    pred_traj_gt_pos[:, start:end]]).view(-1, 2).mean(dim=0, keepdim=True).repeat(
                scene_size, 1)
            szene_origin.append(tmp_origin)
        szene_origin = torch.cat(szene_origin, dim=0)
        return szene_origin

    @staticmethod
    def transform_1(data):

        aug = AugmentSelection.random()
        M = aug.affine()
        data_obs, data_pred_gt = data
        data_obs = Transformer.traj_matmul(data_obs, M)
        data_pred_gt = Transformer.traj_matmul(data_pred_gt, M)
        return data_obs, data_pred_gt

    @staticmethod
    def transform_2(data, seq_start_end):
        # test = torch.rand(1)
        theta = torch.rand(1) * np.pi * 2.
        data_obs, data_pred_gt = data
        data_obs = torch.from_numpy(data_obs).permute(2, 0, 1)
        data_pred_gt = torch.from_numpy(data_pred_gt).permute(2, 0, 1)
        scene_orig = Transformer.scene_origin(data_obs, data_pred_gt, seq_start_end)
        data_obs,_ = Transformer.rotation_2d_torch(data_obs, theta, scene_orig)
        data_pred_gt,_ = Transformer.rotation_2d_torch(data_pred_gt, theta, scene_orig)
        return data_obs.permute(1, 2, 0).numpy(), data_pred_gt.permute(1, 2, 0).numpy()