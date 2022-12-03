# CUDA_VISIBLE_DEVICES=2 python ../inference.py\
#     --model_name gpt2-xl\
#     --data_path ../../../data/human_annotations/human.jsonl\
#     --data_name human_annotations\
#     --decoding_method nucleus\
#     --number_of_instance_to_generate_per_method 1\
#     --prefix_len 32\
#     --decoding_len 256\
#     --save_path_prefix ../inference_results/
# exit

topps=(0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0)

for topp in ${topps[@]}
do
    echo "========== Topp: $topp =========="
    CUDA_VISIBLE_DEVICES=5 python ../inference.py\
        --model_name gpt2-xl\
        --data_path ../../../data/story/story.jsonl\
        --data_name story\
        --decoding_method nucleus\
        --topp $topp\
        --number_of_instance_to_generate_per_method 1\
        --prefix_len 40\
        --decoding_len 200\
        --save_path_prefix ../inference_results/
done
exit

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
