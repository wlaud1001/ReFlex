source_prompt='a man sitting on a rock with trees in the background'
target_prompt='a man sitting on a rock with a city in the background'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 2 \
    --seed 0 \
    --img_path 'data/images/man_tree.jpg' \
    --mask_path 'data/masks/man_tree.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/man_tree' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
