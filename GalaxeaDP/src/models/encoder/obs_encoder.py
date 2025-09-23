import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import os
import numpy as np
import timm
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from src.utils.pytorch_utils import dict_apply
from src.utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class ResNetImageEncoder(nn.Module):
    def __init__(
        self, 
        shape_meta,
        vision_obs_step: int, 
        state_obs_step: int, 
        share_encoder: bool,
        state_mlp_size=(64, 128), 
        state_mlp_activation_fn=nn.ReLU,
        fusion_mlp_size=(512, 256, 128), 
        fusion_mlp_activation_fn=nn.ReLU,
        additional_convs_channel=(128, 32, 8), 
        additional_convs_kernel_size=[1, 1, 1], 
        additional_convs_stride=[1, 1, 1], 
        additional_convs_padding=[0, 0, 0],
    ):
        super().__init__()
        self.vision_obs_step = vision_obs_step
        self.state_obs_step = state_obs_step
        self.state_key = 'qpos'
        self.cams_key = [key for key in shape_meta["obs"] if "rgb" in key]
        self.this_encoder = nn.ModuleDict()
        self.share_encoder = share_encoder
        obs_shape_meta = shape_meta['obs']
        observation_space = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # Resnet18 encoder
        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18_encoder = nn.Sequential(*list(resnet18.children())[:-2])
        additional_convs = nn.Sequential(
            nn.Conv2d(512, additional_convs_channel[0], kernel_size=additional_convs_kernel_size[0], stride=additional_convs_stride[0], padding=additional_convs_padding[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(additional_convs_channel[0], additional_convs_channel[1], kernel_size=additional_convs_kernel_size[1], stride=additional_convs_stride[1], padding=additional_convs_padding[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(additional_convs_channel[1], additional_convs_channel[2], kernel_size=additional_convs_kernel_size[2], stride=additional_convs_stride[2], padding=additional_convs_padding[2]),
            nn.ReLU(inplace=True)
        )
        resnet_encoder = nn.Sequential(
            resnet18_encoder,
            additional_convs
        )
        rgb_encoder = resnet_encoder
        
        image_shape = obs_shape_meta[self.cams_key[0]]['transform_shape']
        test_input = torch.zeros((self.vision_obs_step, 3)+tuple(image_shape))
        test_output = rgb_encoder(test_input)
        test_output = self.aggregate_feature(test_output)
        vision_feature_dim = test_output.shape[-1] * len(self.cams_key)
        
        if share_encoder:
            self.rgb_encoder = rgb_encoder
        else:
            for cam_name in self.cams_key:
                this_model = copy.deepcopy(rgb_encoder)  
                self.this_encoder[cam_name] = this_model

        state_mlp_output_size = 0
        if len(state_mlp_size) == 0:
            log.info(f"ResNetImageEncoder State mlp size is empty")
            self.state_mlp = None
        else:
            if len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            state_shape = observation_space[self.state_key]
            self.state_mlp = nn.Sequential(*create_mlp(state_shape[0], state_mlp_size[-1], net_arch, state_mlp_activation_fn))
            state_mlp_output_size = state_mlp_size[-1]

        fusion_input_dim = vision_feature_dim * self.vision_obs_step + state_mlp_output_size * self.state_obs_step
        if len(fusion_mlp_size) == 0:
            log.info(f"ResNetImageEncoder Fusion mlp size is empty")
            self.fusion_mlp = None
            self.n_output_channels = fusion_input_dim
        else:
            if len(fusion_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = fusion_mlp_size[:-1]
            self.fusion_mlp = nn.Sequential(*create_mlp(fusion_input_dim, fusion_mlp_size[-1], net_arch, fusion_mlp_activation_fn))
            self.n_output_channels = fusion_mlp_size[-1]

    def aggregate_feature(self, feature):
        bs = feature.shape[0]
        feature = feature.reshape(bs, -1)
        return feature

    def forward(self, observations: Dict) -> torch.Tensor:
        if not self.share_encoder:
            rgb_feats = []
            for cam_name in self.cams_key:
                rgb = observations[cam_name] # (B, T, 3, H, W)
                B = rgb.shape[0]
                assert self.vision_obs_step == rgb.shape[1]
                rgb = rgb.flatten(0,1)      
                cur = self.this_encoder[cam_name](rgb)
                cur = self.aggregate_feature(cur)
                rgb_feats.append(cur)  
            rgb_feats = torch.cat(rgb_feats, dim=-1)
        else:
            rgb_list = []
            cam_num = len(self.cams_key)
            for cam_name in self.cams_key:
                rgb = observations[cam_name] # (B, T, 3, H, W)
                B = rgb.shape[0]
                rgb = rgb.flatten(0,1)
                rgb_list.append(rgb)
            rgb_feat = torch.cat(rgb_list, axis=0) #cbt 3 h w
            rgb_feat = self.rgb_encoder(rgb_feat)
            btc, c, h, w = rgb_feat.shape
            bt = btc // cam_num
            rgb_feat = rgb_feat.reshape(cam_num, bt, c, h, w).permute(1, 0, 2, 3, 4).reshape(bt, -1)
            rgb_feats = rgb_feat.reshape(B,-1)
        feats = [rgb_feats]

        if self.state_mlp is not None:
            state = observations[self.state_key].float() # (B, T, state_dim)
            B = state.shape[0]
            assert self.state_obs_step == state.shape[1]
            state = state.flatten(0, 1)
            state_feats = self.state_mlp(state)
            state_feats = state_feats.reshape(B, -1)
            feats.append(state_feats)
        final_feat = torch.cat(feats, dim=-1)
        if self.fusion_mlp is not None:
            final_feat = self.fusion_mlp(final_feat)
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    