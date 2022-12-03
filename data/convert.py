import json
import ipdb

# with open('wikinews/wikinews_contrastive_gpt2-xl_256.jsonl') as f:
with open('wikitext/wikitext_contrastive_gpt2-xl_256.jsonl') as f:
# with open('story/book_contrastive_gpt2-xl_256.jsonl') as f:
    datasets = []
    for idx, line in enumerate(f.readlines()):
        item = json.loads(line)[0]
        new_items = {
            'id': idx, 
            'ended': True,
            'text': item['gold_ref']
        }
        datasets.append(json.dumps(new_items))

# with open('wikinews/wikinews.jsonl', 'w') as f:
with open('wikitext/wikitext.jsonl', 'w') as f:
# with open('story/story.jsonl', 'w') as f:
    for line in datasets:
        f.write(line + '\n')
