CUDA_VISIBLE_DEVICES=0 python ../../inference.py\
    --dataset_path_prefix ../../../data/xsum/\
    --save_path_prefix ../../inference_results/\
    --decoding_len 128\
    --model_name facebook/opt-1.3b\
    --decoding_method resistance\
    --evaluation_mode zero-shot\
    --split_num 1

