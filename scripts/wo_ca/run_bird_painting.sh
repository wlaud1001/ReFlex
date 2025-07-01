source_prompt=''
target_prompt='a photo of an eagle in the sky'

ca_steps=0
sa_steps=12
feature_steps=7

attn_topk=20

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/bird_painting.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/bird_painting' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
