import json
import ipdb

for dataset_name in ['wikitext', 'wikinews', 'story']:
    for method_name in ['greedy', 'topk', 'nucleus', 'contrastive']:
        try:
            with open(f'{dataset_name}/{method_name}/{dataset_name}_{method_name}_gpt2-xl_256.jsonl') as f:
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

            with open(f'{dataset_name}/{method_name}/{method_name}_result.json', 'w') as f:
                json.dump(datasets, f)
        except:
            print(f'[!] not found for {dataset_name}-{method_name}')
