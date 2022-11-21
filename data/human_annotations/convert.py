import json
import ipdb

with open('human_evaluation_annotations.json') as f:
    dataset = json.load(f)

with open('human.jsonl', 'w') as f:
    for idx, item in enumerate(dataset):
        item = {
            'id': idx,
            'ended': True,
            'text': item['prompt']
        }
        line = json.dumps(item)
        f.write(line + '\n')
