# Momentum Decoding For Neural Text Generation
**Authors**: Tian Lan, Yixuan Su, and Shuhang Liu

**[Contact]** If you have any questions, feel free to contact me via (lantiangmftby at gmail.com).

This repository contains code other related resources of our paper ["Momentum Decoding: Open-ended Text Generation As Graph Exploration"](https://arxiv.org/abs/).

****
If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@article{su2022contrastiveiswhatyouneed,
  title={Contrastive Search Is What You Need For Neural Text Generation},
  author={Yixuan Su and Nigel Collier},
  journal={arXiv preprint arXiv:2210.14140},
  year={2022}
}
```

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#inference on benchmarks'>2. Inference on benchmarks</a>
* <a href='#test with prefix'>3. Test with prefix</a>
    
****

<span id='introduction'/>

#### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

Open-ended text generation with autoregressive language models (LMs) is an indispensable component in various NLP applications. Typical examples include dialogue systems , contextual text completion, story generation, etc.

Conventional maximization-based methods for this task, such as greedy search and beam search, often lead to the degeneration problem, i.e. the generated text is unnatural and contains undesirable repetitions.
Existing solutions for this problem can be divided into two categories: 
(1) Stochastic methods, e.g. top-k and nucleus sampling, introduce randomness to avoid undesirable repetitions. However, the intrinsic stochasticity of these sampling approaches often leads to semantic incoherence and topic drift in the generated text.
(2) Deteriminstic method, i.e. contrastive search, relies on a one-step look-ahead mechanism to encourage diverse generations. While obtaining superior performances, such look-ahead operation demands extra computational overhead.

In this study, we perceive open-ended text generation from a new perspective. Specifically, we view it as an exploration process within a directed graph.
Therefore, it allows us to formulate the phenomenon of degeneration as circular loops within the directed graph. In the following figure, we provide an illustration in which the LM generates text given a prefix of three tokens, i.e. [1,2,3], and gets stuck in the circular loops, i.e. repetitions, of [2,3,7,8]. Intuitively, such degeneration can be addressed if the tendency of the LM to stay in the circular loop can be _properly_ discouraged, therefore allowing the LM to jump out of the loop at the correct position and produce text with _natural_ repetitions. Based on this motivation, we propose a novel decoding method---_momentum decoding_---which encourages the LM to greedily explore new nodes outside the current graph. Meanwhile, it also allows the LM to return to the existing nodes but with a momentum downgraded by a pre-defined resistance function. 

<img src="./img/overview.png" width = "350" height = "200" alt="overview" align=center />

Three benchmarks are used in this paper, which are listed under `data` folder (`wikitext`, `wikinews`, `story`).

****


<span id='inference on benchmarks'/>

#### 2. Inference on benchmarks: <a href='#all_catelogue'>[Back to Top]</a>

##### 1. prepare the environment

```bash
pip install -r requirments.txt
```

##### 2. get into the folder

```bash
cd open_ended_generation/english/scripts
```

##### 3. running baselines

The following examples runs on `wikinews` benchmark, replace it with `wikitext` or `story` to test other benchmark

1. run the greedy search

    ```bash
    CUDA_VISIBLE_DEVICES=0 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --data_name wikinews\
    --decoding_method greedy\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```

2. run the beam search
    ```bash
    CUDA_VISIBLE_DEVICES=1 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikinews/wikinews.jsonl\
    --data_name wikinews\
    --decoding_method beam\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```
  
3. run the nucleus sampling

    ```bash
    CUDA_VISIBLE_DEVICES=2 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method nucleus\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```
   
4. run the top-$k$ sampling

    ```bash
    CUDA_VISIBLE_DEVICES=3 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method topk\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```
    
5. run the contrastive search

    ```bash
    CUDA_VISIBLE_DEVICES=6 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --data_name wikitext\
    --decoding_method contrastive\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```
    
6. run the momentum decoding

    ```bash
    CUDA_VISIBLE_DEVICES=7 python ../inference.py\
    --model_name gpt2-xl\
    --data_path ../../../data/story/story.jsonl\
    --data_name story\
    --decoding_method resistance\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/
    ```
    
    
##### 4. test the diversity, MAUVE, and gen-length

This example test the results generated by contrastive search
```bash
CUDA_VISIBLE_DEVICES=1 python ../measure_diversity_mauve_gen_length.py\
    --test_path ../inference_results/gpt2-xl/story/contrastive/contrastive_result.json
```


##### 5. test the Coherence
This example test the results generated by momentum decoding
```bash
CUDA_VISIBLE_DEVICES=6 python ../compute_coherence.py\
    --opt_model_name facebook/opt-2.7b\
    --test_path ../inference_results/gpt2-xl/story/resistance/resistance_result.json
```


##### 6. compute the greedy ratio

```bash
CUDA_VISIBLE_DEVICES=6 python ../compute_greedy_ratio.py \
    --data_path ../../../data/wikitext/wikitext.jsonl\
    --model_name gpt2-xl\
    --data_name wikitext\
    --decoding_method greedy
```

****

<span id='test with prefix'/>

#### 3. Test with prefix: <a href='#all_catelogue'>[Back to Top]</a>

In this example, you will test the Chinese language model [IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese](https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese) with your input after `Prefix >>>`.
If you want to change the langauge model in `test.py`, feel free to change this few lines to load your LMs:

```python
# Change the name in the huggingface models
model = SimCTGGPT('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
model.eval().cuda()
tokenizer = model.tokenizer
# Note, some LMs doesn't have the `eos_token_id`, otherwise the `sep_token_id`
eos_token_id = tokenizer.eos_token_id
```

This example allows you to input your prefix and generate maximum of 512 tokens or achieves the `eos_token_id`.

```bash
./test.sh
```

In this chinese example, if the input prefix is `腾讯是一家`, the momentum decoding and contrastive search will generate the following results:

```text
Prefix >>> 腾讯是一家
[Momentum Decoding] 腾讯是一家非常有活力的公司，我们在移动互联网上也有很多创新，这些创新都是基于对用户需求的深刻理解。” 另外，腾讯还表示，将会与合作伙伴一起，把更多的创新应用带给用户，并通过开放、协作的方式，与更多合作伙伴共同推动中国互联网的发展。
==================================================
[Contrastive Search] 腾讯是一家以互联网为基础的科技与文化公司，专注于移动互联网与社交网络的产品研发与运营。腾讯的使命是“让互联网生活更简单” ，希望通过开放合作的态度来构建一个创新、开放的平台，为各行各业创造价值。 公司简介 腾讯是一家以互联网为基础的科技与文化公司，专注于移动互联网与社交网络的产品研发与运营。腾讯的使命是“让互联网生活更简单” ，希望通过开放合作的态度来构建一个创新、开放的平台，为各行各业创造价值 。 [1] 业务介绍 腾讯拥有国内最大的互联网综合服务平台和行业领先的互联网应用，旗下拥有QQ、
==================================================
```
Note that in this case, the resistance function of momentum decoding is a little bit different from the definition in our paper, the details can be found in the function `` of `models/utils_ngram.py`. This modification is made for this Chinese langauge model, the new langauge models could benefit from the careful definition of resistance function. The resistance function in our paper is made for the English [GPT-XL model](https://huggingface.co/gpt2-xl).

If your test the famous English [gpt2-large](https://huggingface.co/gpt2-large) model, given the prefix `DeepMind Company is`, the generated results of momentum decoding and contrastive search will be:

```text
Prefix >>> DeepMind Company is
[Momentum Decoding] DeepMind Company is a leading AI research company, with over $1.5 billion in funding from investors including Google, Microsoft, and Facebook.

The company's DeepMind AlphaGo program beat the world champion of Go, Lee Sedol, by beating him 6-3 in a match that lasted more than three hours. The game was played on Sunday at the end of a two-day tournament in Seoul.

"We are very proud of our team's achievement," said DeepMind CEO Demis Hassabis. "This is an important step forward for artificial intelligence and we look forward to continuing to work with other companies to develop new technologies."

In addition to its DeepMind subsidiary, DeepMind also has a number of other businesses, including a robotics division, a cloud computing division, and a deep learning startup.
==================================================
[Contrastive Search] DeepMind Company is a leader in artificial intelligence (AI). We have a long history of working with big data, machine learning, deep learning and other areas that have the potential to revolutionize the way we live, work and play.

As part of our mission, we are committed to the open source community and encourage everyone to contribute to our code, research and development. This is the best way to ensure that AI is advancing at breakneck speed and that the benefits of AI are widely shared.

In the past year and a half, we have made significant progress in the areas of AI and machine learning. Our research team has published over 1,000 papers in AI and machine learning journals, and more than 100 of these papers have been accepted for publication in peer-reviewed journals. This progress has been driven by an active and engaged community of researchers, and the support of thousands of contributors who have helped us build this community over the past year and a half.

We believe that AI is going to play a critical role in our future, and it is our responsibility to make sure that it is well-funded, well-staffed and well-funded in every way possible. That's why we are investing in research and development, and in our community, to accelerate the pace of AI development and adoption.

The OpenAI Foundation was founded in 2014 by Elon Musk, Reid Hoffman, Peter Norvig of Google DeepMind, Ilya Sutskever of Facebook AI Research, Chris Ferrucci of Microsoft Research, Yann LeCun of the University of Montreal (Canada), Andrew Ng of Nanyang Technological University (Singapore), and a group of like-minded individuals. The goal of the OAF is to provide a home for AI-related philanthropy and open source software that is free of charge to the public, and to foster the development of AI and machine learning technologies for the benefit of humanity. For more information, visit openai.org.
==================================================
```
