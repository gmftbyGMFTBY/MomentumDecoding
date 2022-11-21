CUDA_VISIBLE_DEVICES=6 python ../compute_greedy_ratio.py \
    --data_path ../../../data/story/story.jsonl\
    --model_name gpt2-xl\
    --data_name story\
    --decoding_method resistance
