source_prompt=''
target_prompt='a photo of a golden statue in a temple'

ca_steps=0
sa_steps=12
feature_steps=7

attn_topk=20

python img_edit.py \
    --gpu 1 \
    --seed 10 \
    --img_path 'data/images/meditation.png' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/meditation' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
