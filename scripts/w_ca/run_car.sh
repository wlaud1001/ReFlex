source_prompt='a sports car driving down the street'
target_prompt='stained glass window of a sports car driving down the street'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=10

python img_edit.py \
    --gpu 1 \
    --seed 0 \
    --img_path 'data/images/car.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/car' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --use_mask 0 \
    --attn_topk $attn_topk
