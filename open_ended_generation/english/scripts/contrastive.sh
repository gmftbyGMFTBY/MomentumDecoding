CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-large\
    --data_path ../../../data/webtext/webtext.test.jsonl\
    --decoding_method contrastive\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
