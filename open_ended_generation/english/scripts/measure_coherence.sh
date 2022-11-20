CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
    --opt_model_name facebook/opt-125m\
    --test_path ../inference_results/gpt2-large/resistance/resistance_result.json
