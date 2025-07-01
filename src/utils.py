import base64
import difflib
import json
import os

import diffusers
import numpy as np
import requests
import torch
import torch.nn.functional as F
import transformers
from diffusers import (AutoencoderKL, DiffusionPipeline,
                       FlowMatchEulerDiscreteScheduler, FluxPipeline,
                       FluxTransformer2DModel, SD3Transformer2DModel,
                       StableDiffusion3Pipeline)
from diffusers.callbacks import PipelineCallback
from torchao.quantization import int8_weight_only, quantize_
from torchvision import transforms
from transformers import (AutoModelForCausalLM, AutoProcessor, CLIPTextModel,
                          CLIPTextModelWithProjection, T5EncoderModel)


def get_flux_pipeline(
    model_id="black-forest-labs/FLUX.1-dev",
    pipeline_class=FluxPipeline,
    torch_dtype=torch.bfloat16, 
    quantize=False
):
    ############ Diffusion Transformer ############
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch_dtype
    )

    ############ Text Encoder ############
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch_dtype
    )

    ############ Text Encoder 2 ############
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype
    )

    ############ VAE ############
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch_dtype
    )

    if quantize:
        quantize_(transformer, int8_weight_only())
        quantize_(text_encoder, int8_weight_only())
        quantize_(text_encoder_2, int8_weight_only())
        quantize_(vae, int8_weight_only())

    # Initialize the pipeline now.
    pipe = pipeline_class.from_pretrained(
        model_id, 
        transformer=transformer, 
        vae=vae,
        text_encoder=text_encoder, 
        text_encoder_2=text_encoder_2, 
        torch_dtype=torch_dtype
    )
    return pipe

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

def mask_interpolate(mask, size=128):
    mask = torch.tensor(mask)
    mask = F.interpolate(mask[None, None, ...], size, mode='bicubic')
    mask = mask.squeeze()
    return mask    

def get_blend_word_index(prompt, word, tokenizer):
    input_ids = tokenizer(prompt).input_ids
    blend_ids = tokenizer(word, add_special_tokens=False).input_ids

    index = []
    for i, id in enumerate(input_ids):
        # Ignore common token
        if id < 100:
            continue
        if id in blend_ids:
            index.append(i)
            
    return index

def find_token_id_differences(prompt1, prompt2, tokenizer):
    # Tokenize inputs and get input IDs
    tokens1 = tokenizer.encode(prompt1, add_special_tokens=False)
    tokens2 = tokenizer.encode(prompt2, add_special_tokens=False)
    
    # Get sequence matcher output
    seq_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    diff1_indices, diff1_ids = [], []
    diff2_indices, diff2_ids = [], []

    for opcode, a_start, a_end, b_start, b_end in seq_matcher.get_opcodes():
        if opcode in ['replace', 'delete']:
            diff1_indices.extend(range(a_start, a_end))
            diff1_ids.extend(tokens1[a_start:a_end])
        if opcode in ['replace', 'insert']:
            diff2_indices.extend(range(b_start, b_end))
            diff2_ids.extend(tokens2[b_start:b_end])

    return {
        'prompt_1': {'index': diff1_indices, 'id': diff1_ids},
        'prompt_2': {'index': diff2_indices, 'id': diff2_ids}
    }

def find_word_token_indices(prompt, word, tokenizer):
    # Tokenize with offsets to track word positions
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.tokens()
    offsets = encoding.offset_mapping  # Start and end positions of tokens in the original text

    word_indices = []
    
    # Normalize the word for comparison
    word_tokens = tokenizer(word, add_special_tokens=False).tokens()

    # Find matching token sequences
    for i in range(len(tokens) - len(word_tokens) + 1):
        if tokens[i : i + len(word_tokens)] == word_tokens:
            word_indices.extend(range(i, i + len(word_tokens)))

    return word_indices