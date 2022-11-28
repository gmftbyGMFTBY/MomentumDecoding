# CUDA_VISIBLE_DEVICES=7 python ../inference.py\
#     --model_name gpt2-xl\
#     --data_path ../../../data/human_annotations/human.jsonl\
#     --data_name human_annotations\
#     --decoding_method resistance\
#     --prefix_len 32\
#     --decoding_len 256\
#     --save_path_prefix ../inference_results/

topks=(2 3 4 5 6 7 8 9 10)
for topk in ${topks[@]}
do
    echo "========== Topk: $topk =========="
    CUDA_VISIBLE_DEVICES=7 python ../inference.py\
        --model_name gpt2-xl\
        --data_path ../../../data/wikinews/wikinews.jsonl\
        --data_name wikinews\
        --decoding_method resistance\
        --prefix_len 40\
        --decoding_len 200\
        --topk $topk\
        --save_path_prefix ../inference_results/\
        --resistance_function ours
done

exit


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
