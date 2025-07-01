import abc
import types

import torch
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock, FluxTransformerBlock)

from .flux_transformer_forward import (joint_transformer_forward,
                                       single_transformer_forward)


class FeatureCollector:
    def __init__(self, transformer, controller, layer_list=[]):
        self.transformer = transformer
        self.controller = controller
        self.layer_list = layer_list

    def register_transformer_control(self):
        index = 0
        for joint_transformer in self.transformer.transformer_blocks:
            place_in_transformer = f'joint_{index}'
            joint_transformer.forward = joint_transformer_forward(joint_transformer, self.controller, place_in_transformer)
            index +=1
            
        for i, single_transformer in enumerate(self.transformer.single_transformer_blocks):
            place_in_transformer = f'single_{index}'
            single_transformer.forward = single_transformer_forward(single_transformer, self.controller, place_in_transformer)
            index +=1

        self.controller.num_layers = index

    def restore_orig_transformer(self):
        place_in_transformer=''
        
        for joint_transformer in self.transformer.transformer_blocks:
            joint_transformer.forward = joint_transformer_forward(joint_transformer, None, place_in_transformer)

        for i, single_transformer in enumerate(self.transformer.single_transformer_blocks):
            single_transformer.forward = single_transformer_forward(single_transformer, None, place_in_transformer)


class FeatureControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_layers = -1
        self.cur_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, place_in_transformer: str):
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, hidden_state, place_in_transformer: str):
        hidden_state = self.forward(hidden_state, place_in_transformer)
        self.cur_layer = self.cur_layer + 1

        if self.cur_layer == self.num_layers:
            self.cur_layer = 0
            self.cur_step = self.cur_step + 1
            self.between_steps()

        return hidden_state

    def reset(self):
        self.cur_step = 0
        self.cur_layer = 0


class FeatureReplace(FeatureControl):
    def __init__(
        self, 
        layer_list=[],
        feature_steps=7
    ):
        super(FeatureReplace, self).__init__()
        self.layer_list = layer_list
        self.feature_steps = feature_steps

    
    def forward(self, hidden_states, place_in_transformer):
        layer_index = int(place_in_transformer.split('_')[-1])
        if (layer_index not in self.layer_list) or (self.cur_step not in range(0, self.feature_steps)):
            return hidden_states

        hs_dim = hidden_states.shape[1]

        t5_dim = 512
        latent_dim = 4096
        attn_dim = t5_dim + latent_dim
        index_all = torch.arange(attn_dim)
        t5_index, latent_index = index_all.split([t5_dim, latent_dim])

        if 'single' in place_in_transformer:
            mask = torch.ones(hs_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)
            mask[t5_index] = 0 # Only use image latent
        else:
            mask = torch.ones(hs_dim).to(device=hidden_states.device, dtype=hidden_states.dtype)

        mask = mask[None, :, None]
        
        source_hs = hidden_states[:1]
        target_hs = hidden_states[1:]
        
        target_hs = source_hs * mask + target_hs * (1 - mask)
        hidden_states[1:] = target_hs
        return hidden_states
