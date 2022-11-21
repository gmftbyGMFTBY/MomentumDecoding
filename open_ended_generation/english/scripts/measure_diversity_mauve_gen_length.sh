datasets=("story" "wikitext" "wikinews")
methods=("topk" "nucleus" "greedy" "contrastive" "resistance")
for dataset in ${datasets[@]}
do
    for method in ${methods[@]}
    do
        CUDA_VISIBLE_DEVICES=1 python ../measure_diversity_mauve_gen_length.py\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result.json
    done
done
