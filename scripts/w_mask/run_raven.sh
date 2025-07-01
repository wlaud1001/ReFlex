source_prompt='a black raven sits on a tree stump in the rain'
target_prompt='a white raven sits on a tree stump in the rain'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 2 \
    --seed 0 \
    --img_path 'data/images/raven.jpg' \
    --mask_path 'data/masks/raven.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/raven' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
