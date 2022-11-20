CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --test_path ../HumanEval.jsonl\
    --decoding_len 128\
    --run_num 1\
    --evaluation_method greedy\
    --model_name Salesforce/codegen-350M-mono\
    --save_path_prefix ../inference_results/
