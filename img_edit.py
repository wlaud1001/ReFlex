import argparse
import gc
import os
import random
import re
import time
from distutils.util import strtobool

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument(
    "--target_prompt",
    type=str,
)
parser.add_argument(
    "--source_prompt",
    type=str,
    default=''
)
parser.add_argument(
    "--blend_word",
    type=str,
    default=''
)
parser.add_argument(
    "--mask_path",
    type=str,
    default=None
)


parser.add_argument(
    "--gpu",
    type=str,
    default="0",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0
)
parser.add_argument(
    "--results_dir",
    type=str,
    default='results'
)


parser.add_argument(
    "--model",
    type=str,
    default='flux',
    choices=['flux']
)

parser.add_argument(
    "--ca_steps",
    type=int,
    default=10,
    help="Number of steps to apply I2T-CA adaptation and injection.",
)

parser.add_argument(
    "--sa_steps",
    type=int,
    default=7,
    help="Number of steps to apply I2I-SA adaptation and injection.",
)

parser.add_argument(
    "--feature_steps",
    type=int,
    default=5,
    help="Number of steps to inject residual features.",
)


parser.add_argument(
    "--ca_attn_layer_from",
    type=int,
    default=13,
    help="Layers to apply I2T-CA adaptation and injection.",
)
parser.add_argument(
    "--ca_attn_layer_to",
    type=int,
    default=45,
    help="Layers to apply I2T-CA adaptation and injection.",
)

parser.add_argument(
    "--sa_attn_layer_from",
    type=int,
    default=20,
    help="Layers to apply I2I-SA adaptation and injection.",
)
parser.add_argument(
    "--sa_attn_layer_to",
    type=int,
    default=45,
    help="Layers to apply I2I-SA adaptation and injection.",
)

parser.add_argument(
    "--feature_layer_from",
    type=int,
    default=13,
    help="Layers to inject residual features.",
)
parser.add_argument(
    "--feature_layer_to",
    type=int,
    default=20,
    help="Layers to inject residual features.",
)

parser.add_argument(
    "--flow_steps",
    type=int,
    default=7,
    help="Steps to apply forward step before inversion",
)
parser.add_argument(
    "--step_start",
    type=int,
    default=0
)


parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=28
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=3.5,
)

parser.add_argument(
    "--attn_topk",
    type=int,
    default=20,
    help="Hyperparameter for I2I-SA adaptaion."
)

parser.add_argument(
    "--text_scale",
    type=float,
    default=4,
    help="Hyperparameter for I2T-CA adaptaion."
)

parser.add_argument(
    "--mid_step_index",
    type=int,
    default=14,
    help="Hyperparameter for mid-step feature extraction."
)


parser.add_argument(
    "--use_mask",
    type=strtobool,
    default=True
)

parser.add_argument(
    "--use_ca_mask",
    type=strtobool,
    default=True
)

parser.add_argument(
    "--mask_steps",
    type=int,
    default=18,
    help="Steps to apply latent blending"
)

parser.add_argument(
    "--mask_dilation",
    type=int,
    default=3
)
parser.add_argument(
    "--mask_nbins",
    type=int,
    default=128
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from src.attn_utils.attn_utils import AttentionAdapter, AttnCollector
from src.attn_utils.flux_attn_processor import NewFluxAttnProcessor2_0
from src.attn_utils.seq_aligner import get_refinement_mapper
from src.callback.callback_fn import CallbackAll
from src.inversion.inverse import get_inversed_latent_list
from src.inversion.scheduling_flow_inverse import \
    FlowMatchEulerDiscreteForwardScheduler
from src.pipeline.flux_pipeline import NewFluxPipeline
from src.transformer_utils.transformer_utils import (FeatureCollector,
                                                     FeatureReplace)
from src.utils import (find_token_id_differences, find_word_token_indices,
                       get_flux_pipeline, mask_decode, mask_interpolate)


def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main(args):
    fix_seed(args.seed)
    device = torch.device('cuda')

    pipe = get_flux_pipeline(
        model_id='black-forest-labs/FLUX.1-dev',
        pipeline_class=NewFluxPipeline
    )
    attn_proc = NewFluxAttnProcessor2_0
    pipe = pipe.to(device)

    layer_order = range(57)

    ca_layer_list = layer_order[args.ca_attn_layer_from:args.ca_attn_layer_to]
    sa_layer_list = layer_order[args.feature_layer_to:args.sa_attn_layer_to]
    feature_layer_list = layer_order[args.feature_layer_from:args.feature_layer_to]

    
    img_path = args.img_path
    source_img = Image.open(img_path).resize((1024, 1024)).convert("RGB")    
    img_base_name = os.path.splitext(img_path)[0].split('/')[-1]
    result_img_dir = f"{args.results_dir}/seed_{args.seed}/{args.target_prompt}"
        
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    prompts = [source_prompt, target_prompt]

    print(prompts)
    mask = None

    if args.use_mask:
        use_mask = True

        if args.mask_path is not None:
            mask = Image.open(args.mask_path)
            mask = torch.tensor(np.array(mask)).bool()
            mask = mask.to(device)

            # Increase the latent blending steps if the ground truth mask is used.
            args.mask_steps = int(args.num_inference_steps * 0.9)
            
            source_ca_index = None
            target_ca_index = None
            use_ca_mask = False

        elif args.use_ca_mask and source_prompt:
            mask = None
            if args.blend_word and args.blend_word in source_prompt:
                editing_source_token_index = find_word_token_indices(source_prompt, args.blend_word, pipe.tokenizer_2)
                editing_target_token_index = None
            else:
                editing_tokens_info = find_token_id_differences(*prompts, pipe.tokenizer_2)
                editing_source_token_index = editing_tokens_info['prompt_1']['index']
                editing_target_token_index = editing_tokens_info['prompt_2']['index']

            use_ca_mask = True
            if editing_source_token_index:
                source_ca_index = editing_source_token_index
                target_ca_index = None
            elif editing_target_token_index:
                source_ca_index = None
                target_ca_index = editing_target_token_index
            else:
                source_ca_index = None
                target_ca_index = None
                use_ca_mask = False

        else:
            source_ca_index = None
            target_ca_index = None
            use_ca_mask = False

    else:
        use_mask = False
        use_ca_mask = False
        source_ca_index = None
        target_ca_index = None

    if source_prompt:
        # Use I2T-CA injection
        mappers, alphas = get_refinement_mapper(prompts, pipe.tokenizer_2, max_len=512)
        mappers = mappers.to(device=device)
        alphas = alphas.to(device=device, dtype=pipe.dtype)
        alphas = alphas[:, None, None, :]

        ca_steps = args.ca_steps
        attn_adj_from = 1

    else:
        # Not use I2T-CA injection
        mappers = None
        alphas = None

        ca_steps = 0
        attn_adj_from=3

    sa_steps = args.sa_steps
    feature_steps = args.feature_steps

    attn_controller = AttentionAdapter(
        ca_layer_list=ca_layer_list,
        sa_layer_list=sa_layer_list,
        ca_steps=ca_steps,
        sa_steps=sa_steps,
        method='replace_topk',
        topk=args.attn_topk,
        text_scale=args.text_scale,
        mappers=mappers,
        alphas=alphas,
        attn_adj_from=attn_adj_from,
        save_source_ca=source_ca_index is not None,
        save_target_ca=target_ca_index is not None,
    )

    attn_collector = AttnCollector(
        transformer=pipe.transformer, 
        controller=attn_controller, 
        attn_processor_class=NewFluxAttnProcessor2_0, 
    )

    feature_controller = FeatureReplace(
        layer_list=feature_layer_list,
        feature_steps=feature_steps,
    )

    feature_collector = FeatureCollector(
        transformer=pipe.transformer, 
        controller=feature_controller,
    )

    num_prompts=len(prompts)
    
    shape = (1, 16, 128, 128)
    generator = torch.Generator(device=device).manual_seed(args.seed)        
    latents = randn_tensor(shape, device=device, generator=generator)
    latents = pipe._pack_latents(latents, *latents.shape)
    
    attn_collector.restore_orig_attention()
    feature_collector.restore_orig_transformer()

    t0 = time.perf_counter()

    inv_latents = get_inversed_latent_list(
        pipe,
        source_img,
        random_noise=latents,
        num_inference_steps=args.num_inference_steps,
        backward_method="ode",
        use_prompt_for_inversion=False,
        guidance_scale_for_inversion=0,
        prompt_for_inversion='',
        flow_steps=args.flow_steps,
    )

    source_latents = inv_latents[::-1]
    target_latents = inv_latents[::-1]

    attn_collector.register_attention_control()
    feature_collector.register_transformer_control()

    callback_fn = CallbackAll(
        latents=source_latents,
        attn_collector=attn_collector, 
        feature_collector=feature_collector, 
        feature_inject_steps=feature_steps,
        mid_step_index=args.mid_step_index,
        step_start=args.step_start,
        use_mask=use_mask,
        use_ca_mask=use_ca_mask,
        source_ca_index=source_ca_index,
        target_ca_index=target_ca_index,
        mask_kwargs={'dilation': args.mask_dilation},
        mask_steps=args.mask_steps,
        mask=mask,
    )

    init_latent = target_latents[args.step_start]
    init_latent = init_latent.repeat(num_prompts, 1, 1)
    init_latent[0] = source_latents[args.mid_step_index]
    
    os.makedirs(result_img_dir, exist_ok=True)
    pipe.scheduler = FlowMatchEulerDiscreteForwardScheduler.from_config(
        pipe.scheduler.config, 
        step_start=args.step_start,
        margin_index_from_image=0
    )

    attn_controller.reset()
    feature_controller.reset()
    attn_controller.text_scale = args.text_scale
    attn_controller.cur_step = args.step_start
    feature_controller.cur_step = args.step_start

    with torch.no_grad():
        images = pipe(
            prompts,               
            latents=init_latent,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            callback_on_step_end=callback_fn,
            mid_step_index=args.mid_step_index,
            step_start=args.step_start,
            callback_on_step_end_tensor_inputs=['latents'],
        ).images
    
    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s.")

    source_img_path = os.path.join(result_img_dir, f"source.png")
    source_img.save(source_img_path)
    
    for i, img in enumerate(images[1:]):
        target_img_path = os.path.join(result_img_dir, f"target_{i}.png")
        img.save(target_img_path)

    target_text_path = os.path.join(result_img_dir, f"target_prompts.txt")
    with open(target_text_path, 'w') as file:
        file.write(target_prompt + '\n')
    
    source_text_path = os.path.join(result_img_dir, f"source_prompt.txt")
    with open(source_text_path, 'w') as file:
        file.write(source_prompt + '\n')

    images = [source_img] + images

    fs=3
    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=(n*fs, 1*fs))

    for i, img in enumerate(images):
        ax[i].imshow(img)

    ax[0].set_title('source')
    ax[1].set_title(source_prompt, fontsize=7)
    ax[2].set_title(target_prompt, fontsize=7)

    overall_img_path = os.path.join(result_img_dir, f"overall.png")
    plt.savefig(overall_img_path, bbox_inches='tight')
    plt.close()
    
    mask_save_dir = os.path.join(result_img_dir, f"mask")
    os.makedirs(mask_save_dir, exist_ok=True)

    if use_ca_mask:
        ca_mask_path = os.path.join(mask_save_dir, f"mask_ca.png")
        mask_img = Image.fromarray((callback_fn.mask.cpu().float().numpy() * 255).astype(np.uint8)).convert('L')
        mask_img.save(ca_mask_path)

    del inv_latents
    del init_latent
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main(args)