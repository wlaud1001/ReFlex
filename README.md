# ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation

### [ICCV 2025] Official Pytorch implementation of the paper: "ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation" 
by Jimyeon Kim, Jungwon Park, Yeji Song, Nojun Kwak, Wonjong Rhee†.

Seoul National University

[Arxiv](https://arxiv.org/abs/2507.01496)
&emsp;
[Project Page](https://wlaud1001.github.io/ReFlex/)



![main](./images/main_figure.png)

## Setup
```
git clone https://github.com/wlaud1001/ReFlex.git
cd ReFlex

conda create -n reflex python=3.10
conda activate reflex
pip install -r requirements.txt
```

## Run

### Run exmaple
```
python img_edit.py \
    --gpu {gpu} \
    --seed {seed} \
    --img_path {source_img_path} \
    --source_prompt {source_prompt} \
    --target_prompt  {target_prompt} \
    --results_dir {results_dir} \
    --feature_steps {feature_steps} \
    --attn_topk {attn_topk}
```
### Arguments
- --gpu: Index of the GPU to use.
- --seed: Random seed.
- --img_path: Path to the input real image to be edited.
- --mask_path (optional): Path to a ground-truth mask for local editing. 
    - If provided, this mask is used directly. 
    - If omitted, the editing mask is automatically generated from attention maps.
- --source_prompt (optional): Text prompt describing the content of the input image.
    - If provided, mask generation and latent blending will be applied.
    - If omitted, editing proceeds without latent blending.
- --target_prompt: Text prompt describing the desired edited image.
- --blend_word (optional): Word in --source_prompt to guide mask generation via its I2T-CA map.
    -  If omitted, the blend word is automatically inferred by comparing source_prompt and target_prompt.
- --results_dir: Directory to save the output images
### 

### Scripts
We also provide several example scripts in the (./scripts) directory for some use cases and reproducible experiments.
#### Script Categories
- scripts/wo_ca/: Cases where the source prompt is not given. I2T-CA adaptation and latent blending are not applied.
- scripts/w_ca/: Cases where the source prompt is given, and the editing mask for latent blending is automatically generated from the attention map.
- scripts/w_mask/: Cases where a ground-truth mask for local editing is provided and directly used for latent blending.

You can run a script as follows:
```
./scripts/wo_ca/run_bear.sh
./scripts/w_ca/run_bird.sh
./scripts/w_mask/run_cat_hat.sh
```