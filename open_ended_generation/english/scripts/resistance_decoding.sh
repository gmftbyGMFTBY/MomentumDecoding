CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --data_name wikinews\
    --decoding_method resistance\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/

CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method resistance\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/

CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/story/story.jsonl\
    --data_name story\
    --decoding_method resistance\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/
