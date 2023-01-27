CUDA_VISIBLE_DEVICES=0 python ../measure_diversity_mauve_gen_length.py\
    --test_path ../inference_results/gpt2-xl/wikitext/resistance/resistance_result_2000.json
exit

datasets=("wikinews")
# methods=("topk" "nucleus" "greedy" "contrastive" "resistance")
methods=("resistance")
for dataset in ${datasets[@]}
do
    for method in ${methods[@]}
    do
        CUDA_VISIBLE_DEVICES=1 python ../measure_diversity_mauve_gen_length.py\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result_log.json
        CUDA_VISIBLE_DEVICES=1 python ../measure_diversity_mauve_gen_length.py\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result_exp.json
        CUDA_VISIBLE_DEVICES=1 python ../measure_diversity_mauve_gen_length.py\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result_constant.json
    done
done
