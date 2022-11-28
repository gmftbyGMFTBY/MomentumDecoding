CUDA_VISIBLE_DEVICES=0 python ../test_degeneration_max_prob.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --data_name wikinews\
    --decoding_method greedy\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
