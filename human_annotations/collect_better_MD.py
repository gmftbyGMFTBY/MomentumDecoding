import json
import ipdb
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default='MD_versus_CS.json')
    return parser.parse_args()

def parse_num(num_1, num_2):
    res = (num_1 / num_2) * 100
    return round(res, 1)

if __name__ == '__main__':
    args = parse_config()
    data_path = args.f
    with open(data_path) as f:
        data = json.load(f)[0]
    method_list = data["compared_methods"]
    assert method_list[0] == 'momentum_decoding'
    compared_method = method_list[1]
    human_annotations = data["human_annotations"]

    collections = []

    for item in human_annotations:
        domain = item['domain']
        annotations = item["human_annotation"]
        system_a_name, system_b_name = item['system_a_name'], item['system_b_name']
        if system_a_name == 'momentum_decoding':
            if list(annotations.values()) == ['system b is more human-like.'] * 3:
                collections.append(item)
        else:
            if list(annotations.values()) == ['system a is more human-like.'] * 3:
                collections.append(item)
    print(f'[!] find {len(collections)} MD better samples')

with open('CS_better_samples.json', 'w') as f:
    json.dump(collections, f, indent=4)
