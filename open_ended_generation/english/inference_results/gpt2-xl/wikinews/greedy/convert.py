import json
import ipdb

with open('wikinews_greedy_gpt2-xl_256.jsonl') as f:
    datasets = []
    for idx, line in enumerate(f.readlines()):
        item = json.loads(line)[0]
        new_item = {
            'prefix_text': item['prompt'].replace('<|endoftext|>', '').strip(),
            'reference_text': item['gold_ref'][len(item['prompt']):],
            'generated_result': {
                '0': item['gen_text'][len(item['prompt']):]
            }
        }
        datasets.append(new_item)

with open('greedy_result.json', 'w') as f:
    json.dump(datasets, f)
