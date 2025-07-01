source_prompt="the christmas illustration of a santa's laughing face"
target_prompt="the christmas illustration of a santa's angry face"

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/santa.jpg' \
    --mask_path 'data/masks/santa.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/santa' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
