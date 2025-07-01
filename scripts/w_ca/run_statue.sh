source_prompt='photo of a statue in front view'
target_prompt='photo of a statue in side view'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=60

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/statue.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/statue' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk \
    --blend_word 'statue'
