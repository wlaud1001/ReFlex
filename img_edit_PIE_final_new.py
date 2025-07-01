import argparse
import gc
import os
import random
import re
from distutils.util import strtobool

import pandas as pd

parser = argparse.ArgumentParser()
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
    default='results_test'
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
    default=10
)

parser.add_argument(
    "--sa_steps",
    type=int,
    default=7
)

parser.add_argument(
    "--feature_steps",
    type=int,
    default=5
)


parser.add_argument(
    "--ca_attn_layer_from",
    type=int,
    default=13
)
parser.add_argument(
    "--ca_attn_layer_to",
    type=int,
    default=45
)

parser.add_argument(
    "--sa_attn_layer_from",
    type=int,
    default=20
)
parser.add_argument(
    "--sa_attn_layer_to",
    type=int,
    default=45
)

parser.add_argument(
    "--feature_layer_from",
    type=int,
    default=13
)
parser.add_argument(
    "--feature_layer_to",
    type=int,
    default=20
)

parser.add_argument(
    "--flow_steps",
    type=int,
    default=7
)
parser.add_argument(
    "--margin_index_from_noise",
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
    default=40
)

parser.add_argument(
    "--text_scale",
    type=float,
    default=4
)

parser.add_argument(
    "--latent_start_index",
    type=int,
    default=14
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
    default=18
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

from src.attn_utils.attn_utils import AttentionP2PPnP, AttnCollector
from src.attn_utils.flux_attn_processor import NewFluxAttnProcessor2_0
from src.callback.callback_fn import CallbackAll, CallbackLatentStore
from src.inversion.inverse import get_inversed_latent_list
from src.inversion.scheduling_flow_inverse import \
    FlowMatchEulerDiscreteForwardScheduler
from src.pipeline.flux_pipeline import NewFluxPipeline
from src.ptp.seq_aligner import get_refinement_mapper
from src.transformer_utils.transformer_utils import (FeatureCollector,
                                                     FeatureReplace)
from src.utils import (find_token_id_differences, get_blend_word_index,
                       get_flux_pipeline, mask_decode, mask_interpolate)


def main(args):
    device = torch.device('cuda')

    pipe = get_flux_pipeline(pipeline_class=NewFluxPipeline)
    attn_proc = NewFluxAttnProcessor2_0
    pipe = pipe.to(device)

    layer_order = range(57)
    ca_layer_list = layer_order[args.ca_attn_layer_from:args.ca_attn_layer_to]
    sa_layer_list = layer_order[args.sa_attn_layer_from:args.sa_attn_layer_to]
    feature_layer_list = layer_order[args.feature_layer_from:args.feature_layer_to]

    root_dir = "dataset/PIE-bench"
    df = pd.read_json(f"{root_dir}/mapping_file.json").T

    torch.set_grad_enabled(False)
    for i in range(len(df)):
        print(len(df), i)
        row_info = df.iloc[i]
        img_path = os.path.join(root_dir, 'annotation_images', row_info['image_path'])
        source_img = Image.open(img_path).resize((1024, 1024)).convert("RGB")

        source_prompt = row_info['original_prompt'].replace('[','').replace(']','')
        target_prompt = row_info['editing_prompt'].replace('[','').replace(']','')
        editing_instruction = row_info['editing_instruction']
        editing_type_id = row_info['editing_type_id']
        blended_word =row_info['blended_word']
        mask_gt = mask_interpolate(mask_decode(row_info['mask']))
        mask_ratio = mask_gt.sum() / (128*128)

        if int(editing_type_id) not in [2, 3]:
            continue

        prompts = [source_prompt, target_prompt]

        seed = args.seed
        fix_seed(seed)
        
        img_base_name = os.path.splitext(img_path)[0]
        result_dir_name = img_base_name
            
        result_img_dir = os.path.join(
            f"{args.results_dir}",
            f"{args.model}",
            f"seed_{args.seed}",
            f"backward_margin{args.margin_index_from_noise}_flow{args.flow_steps}",
            f"step_{args.attn_inject_steps}_{args.feature_inject_steps}_sa{args.sa_steps}",
            f"latent_start_index_{args.latent_start_index}",
            f"feature_layer_{args.feature_layer_from}_{args.feature_layer_to}",
            f"ca_layer_{args.ca_attn_layer_from}_{args.ca_attn_layer_to}",
            f"sa_layer_{args.sa_attn_layer_from}_{args.sa_attn_layer_to}",
            f"{args.method}_topk_{args.attn_topk}_text_scale{args.text_scale}_adj{args.adjust_topk}",
            f"use_mask_{args.use_mask}_ca_mask_{args.use_ca_mask}_steps_{args.mask_steps}_from_{args.ca_extract_step}_dil_{args.mask_dilation}_nbins_{args.mask_nbins}",
            f"{result_dir_name}"
        )
        overall_img_path = os.path.join(
            result_img_dir,
            f"overall.png"
        )

        if os.path.isfile(overall_img_path):
            print(f"File already exist: {overall_img_path}")
            continue

        if args.use_mask:
            if args.use_ca_mask:
                mask = None
                # if mask_ratio < 1:
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
                mask = mask_gt.to(device)
                source_ca_index = None
                use_ca_mask = False
        else:
            mask = None
            source_ca_index = None
            use_ca_mask = False

        if 'p2p' in args.method:
            mappers, alphas = get_refinement_mapper(prompts, pipe.tokenizer_2, max_len=512)
            mappers = mappers.to(device=device)
            alphas = alphas.to(device=device, dtype=pipe.dtype)
            alphas = alphas[:, None, None, :]
        else:
            mappers = None
            alphas = None

        if args.attn_topk >= 4000:
            use_sa_replace=False
        else:
            use_sa_replace=True
        
        topk=args.attn_topk
        attn_controller = AttentionP2PPnP(
            model_name='flux',
            ca_layer_list=ca_layer_list,
            sa_layer_list=sa_layer_list,
            method='replace_topk',
            topk=topk,
            mappers=mappers,
            alphas=alphas,
            use_sa_replace=use_sa_replace,
            sa_steps=args.sa_steps,
            attn_adj_from=1,
            save_source_ca=use_ca_mask,
            save_target_ca=target_ca_index is not None
        )

        attn_collector = AttnCollector(
            model_name='flux',
            transformer=pipe.transformer, 
            controller=attn_controller, 
            attn_processor_class=NewFluxAttnProcessor2_0, 
        )

        feature_controller = FeatureReplace(
            model_name='flux',
            feature_alpha=args.feature_alpha,
            adain_source=args.adain_source,
            feature_latent_only=args.feature_latent_only,
            inject_feature=args.inject_feature
        )

        feature_collector = FeatureCollector(
            model_name='flux',
            transformer=pipe.transformer, 
            controller=feature_controller,
            layer_list=feature_layer_list
        )
    
        num_prompts=len(prompts)
        
        shape = (1, 16, 128, 128)
        generator = torch.Generator(device=device).manual_seed(seed)        
        latents = randn_tensor(shape, device=device, generator=generator)
        if args.model == 'flux':
            latents = pipe._pack_latents(latents, *latents.shape)
        
        attn_collector.restore_orig_attention()
        feature_collector.restore_orig_transformer()

        callback_latent_store = CallbackLatentStore()
        inv_latents = get_inversed_latent_list(
            pipe,
            source_img,
            random_noise=latents,
            callback_fn=callback_latent_store,
            num_inference_steps=28,
            backward_method="ode",
            model_name=args.model,
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
            attn_inject_steps=args.attn_inject_steps, 
            feature_inject_steps=args.feature_inject_steps,
            latent_start_index=args.latent_start_index,
            step_start=args.margin_index_from_noise,
            use_mask=use_ca_mask,
            use_ca_mask=use_ca_mask,
            source_ca_index=source_ca_index,
            target_ca_index=target_ca_index,
            mask_kwargs={'dilation': args.mask_dilation},
            mask_steps=args.mask_steps,
            mask=mask,
        )

        init_latent = target_latents[args.margin_index_from_noise]

        if args.model == 'flux':
            init_latent = init_latent.repeat(num_prompts, 1, 1)
        else:
            init_latent = init_latent.repeat(num_prompts, 1, 1, 1)
        
        init_latent[0] = source_latents[args.latent_start_index]
        
        os.makedirs(result_img_dir, exist_ok=True)
        pipe.scheduler = FlowMatchEulerDiscreteForwardScheduler.from_config(
            pipe.scheduler.config, 
            margin_index_from_noise=args.margin_index_from_noise,
            margin_index_from_image=0
        )

        attn_controller.reset()
        feature_controller.reset()
        attn_controller.text_scale = args.text_scale
        attn_controller.cur_step = args.margin_index_from_noise
        feature_controller.cur_step = args.margin_index_from_noise

        print(source_prompt)
        print(blended_word)
        with torch.no_grad():
            images = pipe(
                prompts,               
                latents=init_latent,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                callback_on_step_end=callback_fn,
                latent_start_index=args.latent_start_index,
                replace_source_step=True,
                replace_target_step=False,
                step_start=args.margin_index_from_noise,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        
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

        editing_instruction_path = os.path.join(result_img_dir, f"editing_instruction.txt")
        with open(editing_instruction_path, 'w') as file:
            file.write(editing_instruction + '\n')
        
        images = [source_img] + images

        fs=3
        n = len(images)
        fig, ax = plt.subplots(1, n, figsize=(n*fs, 1*fs))

        for i, img in enumerate(images):
            ax[i].imshow(img)

        ax[0].set_title('source')
        ax[1].set_title(source_prompt, fontsize=7)
        ax[2].set_title(target_prompt, fontsize=7)

        plt.savefig(overall_img_path, bbox_inches='tight')
        plt.close()
        
        mask_save_dir = os.path.join(result_img_dir, f"mask")
        os.makedirs(mask_save_dir, exist_ok=True)

        if mask_gt is not None:
            mask_img_path = os.path.join(mask_save_dir, f"mask_gt.png")
            mask_img = Image.fromarray((mask_gt.cpu().numpy() * 255).astype(np.uint8)).convert('L')
            mask_img.save(mask_img_path)
        
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