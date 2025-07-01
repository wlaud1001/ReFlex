import abc
import gc
import math
import numbers
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu


class AttentionControl(abc.ABC):
    def __init__(self,):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        self.get_model_info()
    
    def get_model_info(self):
        t5_dim = 512
        latent_dim = 4096
        attn_dim = t5_dim + latent_dim
        index_all = torch.arange(attn_dim)
        t5_index, latent_index = index_all.split([t5_dim, latent_dim])
        patch_order = ['t5', 'latent']
        
        self.model_info = {
            't5_dim': t5_dim,
            'latent_dim': latent_dim,
            'attn_dim': attn_dim,
            't5_index': t5_index,
            'latent_index': latent_index,
            'patch_order': patch_order
        }

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, q, k, v, place_in_transformer: str):
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, q, k, v, place_in_transformer: str):
        hs = self.forward(q, k, v, place_in_transformer)
    
        self.cur_att_layer += 1
        
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1

            self.between_steps()
        return hs

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight

    def split_attn(self, attn, q='latent', k='latent'):
        patch_order = self.model_info['patch_order']
        t5_dim = self.model_info['t5_dim']
        latent_dim = self.model_info['latent_dim']
        clip_dim = self.model_info.get('clip_dim', None)

        idx_q = patch_order.index(q)
        idx_k = patch_order.index(k)
        split = [t5_dim, latent_dim]
        
        return attn.split(split, dim=-2)[idx_q].split(split, dim=-1)[idx_k].clone()


class AttentionAdapter(AttentionControl):
    def __init__(
        self, 
        ca_layer_list=list(range(13,45)),
        sa_layer_list=list(range(22,45)),
        method='replace_topk',
        topk=1,
        text_scale=1,
        mappers=None,
        alphas=None,
        ca_steps=10,
        sa_steps=7,
        save_source_ca=False,
        save_target_ca=False,
        use_sa_replace=True,
        attn_adj_from=0,
    ):
        super(AttentionAdapter, self).__init__()
        self.ca_layer_list = ca_layer_list
        self.sa_layer_list = sa_layer_list
        self.method = method
        self.topk = topk
        self.text_scale = text_scale
        self.use_sa_replace = use_sa_replace

        self.ca_steps = ca_steps
        self.sa_steps = sa_steps

        self.mappers = mappers
        self.alphas = alphas

        self.save_source_ca = save_source_ca
        self.save_target_ca = save_target_ca

        self.attn_adj_from = attn_adj_from

        self.source_ca = None

        self.source_attn = {}

    @staticmethod
    def get_empty_store():
        return defaultdict(list)
    
    def refine_ca(self, source_ca, target_ca):
        source_ca_replace = source_ca[:, :, self.mappers].permute(2, 0, 1, 3)
        new_ca = source_ca_replace * self.alphas + target_ca * (1 - self.alphas) * self.text_scale
        return new_ca

    def replace_ca(self, source_ca, target_ca):
        new_ca = torch.einsum('hpw,bwn->bhpn', source_ca, self.mappers)
        return new_ca
    
    def get_index_from_source(self, attn, topk):
        if self.method == 'replace_topk':
            sa_max = torch.topk(attn, k=topk, dim=-1)[0][..., [-1]]
            idx_from_source = (attn > sa_max)
        elif self.method == 'replace_z':
            log_attn = torch.log(attn)
            idx_from_source = log_attn > (log_attn.mean(-1, keepdim=True) + self.z_value * log_attn.std(-1, keepdim=True))
        else:
            print("No method")
        return idx_from_source

    def forward(self, q, k, v, place_in_transformer: str):
        layer_index = int(place_in_transformer.split('_')[-1])

        use_ca_replace = False
        use_sa_replace = False
        if (layer_index in self.ca_layer_list) and (self.cur_step in range(0, self.ca_steps)):
            if self.mappers is not None:
                use_ca_replace = True
        if (layer_index in self.sa_layer_list) and (self.cur_step in range(0, self.sa_steps)):
            use_sa_replace = True

        if not (use_sa_replace or use_ca_replace):
            return F.scaled_dot_product_attention(q, k, v)
        
        latent_index = self.model_info['latent_index']
        t5_index = self.model_info['t5_index']
        clip_index = self.model_info.get('clip_index', None)

        # Get attention map first
        attn = self.scaled_dot_product_attention(q, k, v)
        source_attn = attn[:1]
        target_attn = attn[1:]
        source_hs = source_attn @ v[:1] 
            
        source_ca = self.split_attn(source_attn, 'latent', 't5')
        target_ca = self.split_attn(target_attn, 'latent', 't5')

        if use_ca_replace:
            if self.save_source_ca:
                if layer_index == self.ca_layer_list[0]:
                    self.source_ca = source_ca / source_ca.sum(dim=-1, keepdim=True)
                else:
                    self.source_ca += source_ca / source_ca.sum(dim=-1, keepdim=True)

            if self.save_target_ca:
                if layer_index == self.ca_layer_list[0]:
                    self.target_ca = target_ca / target_ca.sum(dim=-1, keepdim=True)
                else:
                    self.target_ca += target_ca / target_ca.sum(dim=-1, keepdim=True)

            if self.alphas is None:
                target_ca = self.replace_ca(source_ca[0], target_ca)
            else:
                target_ca = self.refine_ca(source_ca[0], target_ca)

        target_sa = self.split_attn(target_attn, 'latent', 'latent')
        if use_sa_replace:
            if self.cur_step < self.attn_adj_from:
                topk = 1
            else:
                topk = self.topk

            if self.method == 'base':
                new_sa = self.split_attn(target_attn, 'latent', 'latent')
            else:
                source_sa = self.split_attn(source_attn, 'latent', 'latent')
                if topk <= 1:
                    new_sa = source_sa.clone().repeat(len(target_attn), 1, 1, 1)
                else:
                    idx_from_source = self.get_index_from_source(source_sa, topk)
                    # Get top-k attention values from target SA
                    new_sa = target_sa.clone()
                    new_sa.mul_(idx_from_source)
                    # Normalize
                    new_sa.div_(new_sa.sum(-1,keepdim=True))
                    new_sa.nan_to_num_(0)
                    new_sa.mul_((source_sa * idx_from_source).sum(-1, keepdim=True))
                    # Get rest attention vlaues from source SA
                    new_sa.add_(source_sa * idx_from_source.logical_not())
                    # Additional normalize (Optional)
                    # new_sa.mul_((target_sa.sum(dim=(-1), keepdim=True) / new_sa.sum(dim=(-1), keepdim=True)))
            target_sa = new_sa

        target_l_to_l = target_sa @ v[1:, :, latent_index]
        target_l_to_t = target_ca @ v[1:, :, t5_index]
        
        if self.alphas is None:
            target_latent_hs = target_l_to_l + target_l_to_t * self.text_scale
        else:
            # text scaling is already performed in self.refine_ca()
            target_latent_hs = target_l_to_l + target_l_to_t

        target_text_hs = target_attn[:,:, t5_index,:] @ v[1:]
        target_hs = torch.cat([target_text_hs, target_latent_hs], dim=-2)
        hs = torch.cat([source_hs, target_hs])
        return hs

    def reset(self):
        super(AttentionAdapter, self).reset()
        del self.source_attn
        gc.collect()
        torch.cuda.empty_cache()

        self.source_attn = {}

class AttnCollector:
    def __init__(self, transformer, controller, attn_processor_class, layer_list=[]):
        self.transformer = transformer
        self.controller = controller
        self.attn_processor_class = attn_processor_class

    def restore_orig_attention(self):
        attn_procs = {}
        place=''
        for i, (name, attn_proc) in enumerate(self.transformer.attn_processors.items()):
            attn_proc = self.attn_processor_class(
                controller=None, place_in_transformer=place,
            )
            attn_procs[name] = attn_proc
        self.transformer.set_attn_processor(attn_procs)
        self.controller.num_att_layers = 0
            
    def register_attention_control(self):
        attn_procs = {}
        count = 0
        for i, (name, attn_proc) in enumerate(self.transformer.attn_processors.items()):
            if 'single' in name:
                place = f'single_{i}'
            else:
                place = f'joint_{i}'
            count += 1
            
            attn_proc = self.attn_processor_class(
                controller=self.controller, place_in_transformer=place,
            )
            attn_procs[name] = attn_proc
            
        self.transformer.set_attn_processor(attn_procs)
        self.controller.num_att_layers = count
