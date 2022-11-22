import json
import ipdb

methods = ['contrastive', 'contrastive_decoding', 'topk', 'nucleus', 'resistance']

results = {method: None for method in methods}

for method in methods:
    path = f'human_annotations/{method}/{method}_result.json'
    with open(path) as f:
        data = json.load(f)
        results[method] = data

# re-collect the samples by the sessions
## length sanity check
lengths = [len(results[method]) for method in methods]
for l, method in zip(lengths, methods):
    if l != 150:
        raise Exception(f'[!] {method} not have 150 sessions, but get {l} sessions')

sessions = []
for i in range(150):
    prompts = []
    for m in methods:
        prompts.append(results[m][i]['prefix_text'])
    prompts = set(prompts)
    if len(prompts) > 1:
        raise Exception(f'[!] different prompts for session: {i}')

