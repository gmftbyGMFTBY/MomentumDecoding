
# cuda_visible_devices=6 python ../inference.py\
#     --model_name gpt2-xl\
#     --data_path ../../../data/human_annotations/human.jsonl\
#     --data_name human_annotations\
#     --decoding_method contrastive\
#     --prefix_len 32\
#     --decoding_len 256\
#     --save_path_prefix ../inference_results/
topks=(2 3 4 6 7 8 9 10)
for topk in ${topks[@]}
do
    CUDA_VISIBLE_DEVICES=6 python ../inference.py\
        --model_name gpt2-xl\
        --data_path ../../../data/story/story.jsonl\
        --data_name story\
        --decoding_method contrastive\
        --topk $topk\
        --prefix_len 40\
        --decoding_len 200\
        --save_path_prefix ../inference_results/
done
exit

CUDA_VISIBLE_DEVICES=6 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method contrastive\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/
    
CUDA_VISIBLE_DEVICES=6 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/story/story.jsonl\
    --data_name story\
    --decoding_method contrastive\
    --prefix_len 32\
    --decoding_len 256\
    --save_path_prefix ../inference_results/
