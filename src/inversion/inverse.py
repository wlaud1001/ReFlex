import gc

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torchvision import transforms

from ..callback.callback_fn import CallbackLatentStore
from .scheduling_flow_inverse import (FlowMatchEulerDiscreteBackwardScheduler,
                                      FlowMatchEulerDiscreteForwardScheduler)


@torch.no_grad()
def img_to_latent(img, vae):
    normalize = transforms.Normalize(mean=[0.5],std=[0.5])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    tensor_img = trans(img)[None, ...]
    tensor_img = tensor_img.to(dtype=vae.dtype, device=vae.device)
    posterior = vae.encode(tensor_img).latent_dist
    latents = (posterior.mean - vae.config.shift_factor) * vae.config.scaling_factor
    # latents = posterior.mean
    return latents


@torch.no_grad()
def get_inversed_latent_list(
    pipe, 
    image: Image,
    random_noise=None,
    num_inference_steps: int = 28,
    backward_method: str = 'ode',
    model_name: str = 'flux',
    res=(1024, 1024),
    use_prompt_for_inversion=False,
    guidance_scale_for_inversion=0,
    prompt_for_inversion=None,
    seed=0,
    flow_steps=1,
    ode_steps=1,
    intermediate_steps=None
):
    img = image.resize(res)
    img_latent = img_to_latent(image, pipe.vae)
    device = img_latent.device

    generator = torch.Generator(device=device).manual_seed(seed)


    if random_noise is None:
        random_noise = randn_tensor(img_latent.shape, device=device, generator=generator)
        if model_name == 'flux':
            random_noise = pipe._pack_latents(random_noise, *random_noise.shape)
    if model_name == 'flux':
        img_latent = pipe._pack_latents(img_latent, *img_latent.shape)

    pipe.scheduler = FlowMatchEulerDiscreteBackwardScheduler.from_config(
        pipe.scheduler.config, 
        margin_index_from_noise=0,
        margin_index_from_image=0,
        intermediate_steps=intermediate_steps
    )
    if model_name == 'flux':
        image_seq_len = img_latent.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    else:
        mu = None
        sigmas = None
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, mu=mu, sigmas=sigmas)

    sigmas = pipe.scheduler.sigmas
    timesteps = pipe.scheduler.timesteps

    if backward_method == 'flow':
        inv_latents = [img_latent]
        for sigma in sigmas:
            inv_latent = (1 - sigma) * img_latent + sigma * random_noise
            inv_latents.append(inv_latent)
 
    elif backward_method == 'ode':
        inv_latents = [img_latent]
        img_latent_new = img_latent.to(pipe.dtype)
        random_noise = random_noise.to(pipe.dtype)

        callback_fn = CallbackLatentStore()
        inv_latent = pipe.inversion(
            latents=img_latent_new,
            rand_latents=random_noise,
            flow_steps=flow_steps,
            prompt=prompt_for_inversion if use_prompt_for_inversion else '',
            num_images_per_prompt=1,
            output_type='latent',
            width=res[0], height=res[1],
            guidance_scale=guidance_scale_for_inversion,
            num_inference_steps=num_inference_steps,
            callback_on_step_end=callback_fn
        ).images
        inv_latents = inv_latents + callback_fn.latents
    del img_latent
    gc.collect()
    torch.cuda.empty_cache()

    return inv_latents