source_prompt='an older couple walking down a narrow dirt road'
target_prompt='an older couple walking down a snow coverd road'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 2 \
    --seed 0 \
    --img_path 'data/images/old_couple.jpg' \
    --mask_path 'data/masks/old_couple.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/old_couple' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
