CUDA_VISIBLE_DEVICES=4 python ../../inference.py\
    --dataset_path_prefix ../../../data/xsum/\
    --save_path_prefix ../../inference_results/\
    --decoding_len 128\
    --model_name facebook/opt-350m\
    --decoding_method greedy\
    --evaluation_mode two-shot\
    --split_num 1

