source_prompt=''
target_prompt='an image of Paddington the bear'

ca_steps=0
sa_steps=12
feature_steps=7

attn_topk=20

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/bear.jpeg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/bear' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
    