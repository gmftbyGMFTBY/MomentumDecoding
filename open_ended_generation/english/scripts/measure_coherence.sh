CUDA_VISIBLE_DEVICES=1 python ../compute_coherence.py\
    --opt_model_name facebook/opt-2.7b\
    --test_path ../inference_results/gpt2-xl/story/resistance/resistance_result_2000.json
exit

methods=("resistance")
# datasets=("story" "wikinews" "wikitext")
datasets=("wikinews")
for dataset in ${datasets[@]}
do
    for method in ${methods[@]}
    do
        # CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
        #     --opt_model_name facebook/opt-2.7b\
        #     --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result.json


        CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
            --opt_model_name facebook/opt-2.7b\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result_exp.json
        CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
            --opt_model_name facebook/opt-2.7b\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result_constant.json
    done
done
