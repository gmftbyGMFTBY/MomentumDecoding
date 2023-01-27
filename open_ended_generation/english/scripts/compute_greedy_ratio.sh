CUDA_VISIBLE_DEVICES=4 python ../compute_greedy_ratio.py \
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --model_name gpt2-xl\
    --data_name wikinews\
    --decoding_method resistance
