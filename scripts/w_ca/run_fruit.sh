source_prompt='white plate with fruits on it'
target_prompt='white plate with pizza on it'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=40

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/fruit.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/fruit' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk 