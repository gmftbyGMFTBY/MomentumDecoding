import json
import ipdb

methods = ['contrastive', 'resistance']
datasets = ['story', 'wikitext', 'wikinews']

results = {dataset: {method: None for method in methods } for dataset in datasets}
for dataset in datasets:
    for method in methods:
        path = f'{dataset}/{method}/{method}_result.json'
        with open(path) as f:
            data = json.load(f)
            results[dataset][method] = data

for dataset in datasets:
    assert len(results[dataset]['contrastive']) == len(results[dataset]['resistance'])
    for i in range(len(results[dataset]['contrastive'])):
        cs_prompt = results[dataset]['contrastive'][i]['prefix_text']
        md_prompt = results[dataset]['resistance'][i]['prefix_text']
        assert cs_prompt == md_prompt, f'prefix_text is different at {i}'
