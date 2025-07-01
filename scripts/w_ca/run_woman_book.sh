source_prompt='a woman sitting in the grass with a book'
target_prompt='a woman sitting in the grass with a laptop'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 1 \
    --seed 0 \
    --img_path 'data/images/woman_book.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/woman_book' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk
