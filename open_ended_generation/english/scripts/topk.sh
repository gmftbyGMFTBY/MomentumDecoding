# CUDA_VISIBLE_DEVICES=3 python ../inference.py\
#     --model_name gpt2-xl\
#     --data_path ../../../data/human_annotations/human.jsonl\
#     --data_name human_annotations\
#     --decoding_method topk\
#     --number_of_instance_to_generate_per_method 1\
#     --prefix_len 32\
#     --decoding_len 256\
#     --save_path_prefix ../inference_results/
# exit

topks=(5 10 20 40 50 80 160 320 640)
for topk in ${topks[@]}
do
    echo "========== Topk: $topk =========="
    CUDA_VISIBLE_DEVICES=4 python ../inference.py\
        --model_name gpt2-xl\
        --data_path ../../../data/wikinews/wikinews.jsonl\
        --data_name wikinews\
        --decoding_method topk\
        --number_of_instance_to_generate_per_method 1\
        --topk $topk\
        --prefix_len 40\
        --decoding_len 200\
        --save_path_prefix ../inference_results/
done
exit

CUDA_VISIBLE_DEVICES=3 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method topk\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/

CUDA_VISIBLE_DEVICES=3 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/story/story.jsonl\
    --data_name story\
    --decoding_method topk\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
