methods=("topk" "nucleus")
datasets=("story" "wikinews" "wikitext")
for dataset in ${datasets[@]}
do
    for method in ${methods[@]}
    do
        CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
            --opt_model_name facebook/opt-2.7b\
            --test_path ../inference_results/gpt2-xl/$dataset/$method/${method}_result.json
    done
done
