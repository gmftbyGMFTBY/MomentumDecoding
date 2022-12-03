import json
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

    result_dict = {}
    for domain in ['book', 'wikinews', 'wikitext']:
        result_dict[domain] = {}
        for method in method_list + ['equal']:
            result_dict[domain][method] = 0

    for item in human_annotations:
        domain = item['domain']
        annotations = item["human_annotation"]
        system_a_name, system_b_name = item['system_a_name'], item['system_b_name']
        for key in annotations:
            one_annotation = annotations[key]
            if one_annotation == "system a is more human-like.":
                result_dict[domain][system_a_name] += 1
            elif one_annotation == "system b is more human-like.":
                result_dict[domain][system_b_name] += 1
            elif one_annotation == "two systems are comparable.":
                result_dict[domain]['equal'] += 1
            else:
                raise Exception('Wrong Annotation!!!')

    for domain in ['wikinews', 'wikitext', 'book']:
        if domain == 'book':
            printed_domain = 'Story'
        elif domain == 'wikinews':
            printed_domain = 'Wikinews'
        else:
            printed_domain = 'Wikitext'

        momentum_decoding_result = result_dict[domain]['momentum_decoding']
        compared_method_result = result_dict[domain][compared_method]
        equal_result = result_dict[domain]['equal']
        overall_num = momentum_decoding_result + compared_method_result + equal_result
        momentum_decoding_result = parse_num(momentum_decoding_result, overall_num)
        compared_method_result = parse_num(compared_method_result, overall_num)
        equal_result = round(100 - momentum_decoding_result - compared_method_result, 1)
        
        print ('========================================== Domain: {} =========================================='.format(printed_domain))
        print ('momentum_decoding is better at {}%; Two methods are comparable at {}%; {} is better at {}%'.format(
        momentum_decoding_result, equal_result, compared_method, compared_method_result))
        print ('------------------------------------------------------------------------------------------------------'+'\n')
        
