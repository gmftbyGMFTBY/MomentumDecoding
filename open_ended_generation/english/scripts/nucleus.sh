CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/human_annotations/human.jsonl\
    --data_name human_annotations\
    --decoding_method nucleus\
    --number_of_instance_to_generate_per_method 1\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/
exit


CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --data_name wikinews\
    --decoding_method nucleus\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/

CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method nucleus\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/

CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/story/story.jsonl\
    --data_name story\
    --decoding_method nucleus\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
