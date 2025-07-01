source_prompt='a blue and white bird sits on a branch'
target_prompt='a blue and white butterfly sits on a branch'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 3 \
    --seed 0 \
    --img_path 'data/images/bird.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/bird' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
