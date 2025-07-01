source_prompt='a koala is sitting on a tree'
target_prompt='a koala and a bird is sitting on a tree'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=40

python img_edit.py \
    --gpu 3 \
    --seed 0 \
    --img_path 'data/images/koala.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/koala' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
