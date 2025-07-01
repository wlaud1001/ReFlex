source_prompt='a cat wearing a pink hat'
target_prompt='a tiger wearing a pink hat'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 3 \
    --seed 0 \
    --img_path 'data/images/cat_hat.jpg' \
    --mask_path 'data/masks/cat_hat.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/cat_hat' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
