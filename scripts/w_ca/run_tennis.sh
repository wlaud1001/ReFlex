source_prompt='a woman in a black tank top and pink shorts is about to hit a tennis ball'
target_prompt='a iron woman robot in a black tank top and pink shorts is about to hit a tennis ball'

ca_steps=10
sa_steps=7
feature_steps=5

attn_topk=20

python img_edit.py \
    --gpu 0 \
    --seed 0 \
    --img_path 'data/images/tennis.jpg' \
    --source_prompt "$source_prompt" \
    --target_prompt  "$target_prompt" \
    --results_dir 'results/tennis' \
    --ca_steps $ca_steps \
    --sa_steps $sa_steps \
    --feature_steps $feature_steps \
    --attn_topk $attn_topk \
    --blend_word 'woman'
