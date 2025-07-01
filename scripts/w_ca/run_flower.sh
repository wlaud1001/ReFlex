source_prompt='a pink flower with yellow center in the middle'
target_prompt='a blue flower with red center in the middle'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 1 \
    --seed 0 \
    --img_path 'data/images/flower.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/flower' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk \
    --blend_word 'flower'
