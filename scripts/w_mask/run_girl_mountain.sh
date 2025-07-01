source_prompt='a woman with her arms outstretched on top of a mountain'
target_prompt='a woman with her arms outstretched in front of the NewYork'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 2 \
    --seed 0 \
    --img_path 'data/images/girl_mountain.jpg' \
    --mask_path 'data/masks/girl_mountain.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/girl_mountain' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
