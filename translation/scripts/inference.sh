CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --dataset_path_prefix ../../data/translation/\
    --evaluation_perl_script_path ../\
    --benchmark_name iwslt14\
    --translation_direction de-to-en\
    --save_path_prefix ../inference_results/\
    --decoding_len 128\
    --decoding_method resistance\
    --model_name facebook/opt-125m\
    --shot 1\
    --split_num 3

