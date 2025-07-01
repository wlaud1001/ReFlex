source_prompt='a photo of couples dancing'
target_prompt='a photo of silver robots dancing'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 2 \
    --seed 0 \
    --img_path 'data/images/dancing.jpeg' \
    --mask_path 'data/masks/dancing.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/dancing' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
