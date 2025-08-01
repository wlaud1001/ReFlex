source_prompt='a plate with steak on it'
target_prompt='a plate with salmon on it'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=40

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/steak.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/steak' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
