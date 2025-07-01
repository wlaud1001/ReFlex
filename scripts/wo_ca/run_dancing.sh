source_prompt=''
target_prompt='a couple of silver robots dancing in the garden'

ca_steps=0
sa_steps=12
feature_steps=7

attn_topk=20

python img_edit.py \
    --gpu 3 \
    --seed 0 \
    --img_path 'data/images/dancing.jpeg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/dancing' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk