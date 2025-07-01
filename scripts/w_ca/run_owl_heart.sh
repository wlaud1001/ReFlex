source_prompt='a cartoon painting of a cute owl with a heart on its body'
target_prompt='a cartoon painting of a cute owl with a circle on its body'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 1 \
    --seed 0 \
    --img_path 'data/images/owl_heart.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/owl_heart' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
