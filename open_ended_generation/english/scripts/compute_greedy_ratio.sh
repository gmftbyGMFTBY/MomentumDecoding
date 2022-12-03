CUDA_VISIBLE_DEVICES=6 python ../compute_greedy_ratio.py \
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --model_name gpt2-xl\
    --data_name wikitext\
    --decoding_method greedy
