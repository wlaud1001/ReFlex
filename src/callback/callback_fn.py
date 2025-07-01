import torch
import torch.nn.functional as F
from diffusers.callbacks import PipelineCallback
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu

from ..attn_utils.mask_utils import get_mask


class CallbackLatentStore(PipelineCallback):
    tensor_inputs = ['latents']
    
    def __init__(self):
        self.latents = []

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        self.latents.append(callback_kwargs['latents'])
        return callback_kwargs

class CallbackAll(PipelineCallback):
    tensor_inputs = ['latents']
    def __init__(
        self, 
        latents, 
        attn_collector, 
        feature_collector, 
        feature_inject_steps, 
        mid_step_index=0,
        step_start=0,
        use_mask=False,
        use_ca_mask=False,
        source_ca_index=None,
        target_ca_index=None,
        mask_steps=18,
        mask_kwargs={},
        mask=None,
    ):
        self.latents = latents

        self.attn_collector = attn_collector
        self.feature_collector = feature_collector
        self.feature_inject_steps = feature_inject_steps

        self.mid_step_index = mid_step_index
        self.step_start = step_start
        
        self.mask = mask
        self.mask_steps = mask_steps

        self.use_mask = use_mask
        self.use_ca_mask = use_ca_mask
        self.source_ca_index = source_ca_index
        self.target_ca_index = target_ca_index
        self.mask_kwargs = mask_kwargs

    def latent_blend(self, s, t, mask):
        return s * (1-mask) + t * mask
        # return s * mask.logical_not() + t * mask

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs):
        cur_step = step_index + self.step_start

        if self.latents is None:
            pass
        elif cur_step < self.mid_step_index:
            inject_latent = self.latents[self.mid_step_index]
            callback_kwargs['latents'][:1] = inject_latent

        if self.use_mask:
            if self.use_ca_mask:
                if self.source_ca_index is not None:
                    source_ca = self.attn_collector.controller.source_ca
                    mask = get_mask(source_ca, self.source_ca_index, **self.mask_kwargs)
                elif self.target_ca_index is not None:
                    if cur_step < 5:
                        return callback_kwargs
                    target_ca = self.attn_collector.controller.target_ca
                    mask = get_mask(target_ca, self.target_ca_index, **self.mask_kwargs)
                self.mask = mask
            elif self.mask is not None:
                mask = self.mask
            else:
                return callback_kwargs

            if (cur_step < self.mask_steps):
                mask = mask.to(pipeline.dtype)
                target_latent = callback_kwargs['latents'][1:]
                blend_latent = self.latents[cur_step+1]
                # if cur_step + 1 < self.mid_step_index:
                #     blend_latent = self.latents[cur_step+1]
                # else:
                #     blend_latent = callback_kwargs['latents'][:1]
                
                new_latent = self.latent_blend(
                    pipeline._unpack_latents(blend_latent, 1024, 1024, pipeline.vae_scale_factor), 
                    pipeline._unpack_latents(target_latent, 1024, 1024, pipeline.vae_scale_factor),
                    mask
                )
                new_latent = pipeline._pack_latents(new_latent, *new_latent.shape)
                callback_kwargs['latents'][1:] = new_latent

        return callback_kwargs