import json
import torch
import argparse
import progressbar
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import ipdb

class Data:
    def __init__(self, tokenizer, test_path, data_path, prefix_len, decoding_len, number_of_instance_to_generate_per_method):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.decoding_len = decoding_len
        self.number_of_instance_to_generate_per_method = number_of_instance_to_generate_per_method
        self.prefix_token_id_list = self.get_prefix_file(data_path)
        self.result_token_id_list = self.get_result_file(test_path)
        print('Evaluation number is {}'.format(len(self.prefix_token_id_list)))
    
    def get_prefix_file(self, data_path):
        print ('Get prefix from {}'.format(data_path))
        prefix_token_id_list = []
        
        import json
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        n = len(data)
        print ('Prefix number is {}'.format(n))
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            item = data[i]
            text = item['text']
            token_id_list = self.get_one_prefix(text)
            if token_id_list != []:
                prefix_token_id_list.append(token_id_list)
        return prefix_token_id_list
    
    def get_one_prefix(self, text):
        tokens = self.tokenizer.tokenize(text)
        total_len = self.prefix_len + self.decoding_len
        if len(tokens) < total_len:
            return []
        
        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        prefix_id_list = token_id_list[:self.prefix_len]
        return prefix_id_list

    def get_result_file(self, test_path):
        print ('Get result from {}'.format(test_path))
        result_token_id_list = []

        data = json.load(open(test_path))
        n = len(data)
        print ('Result number is {}'.format(n))
        assert n == len(self.prefix_token_id_list)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            temp_token_id_list = []
            for idx in range(self.number_of_instance_to_generate_per_method):
                temp_token_id_list.append(self.get_one_result(data[i]['generated_result'][str(idx)]))
            result_token_id_list.append(temp_token_id_list)
        p.finish()
        return result_token_id_list

    def get_one_result(self, text):
        result_tokens = self.tokenizer.tokenize(text)
        result_id_list = self.tokenizer.convert_tokens_to_ids(result_tokens)

        return result_id_list
        
def inference_one_instance(args, prefix_token_id_list, source_token_id_list, model, cuda_available, device):
    input_ids = torch.LongTensor(prefix_token_id_list + source_token_id_list).view(1, -1)
    ipdb.set_trace()

    if cuda_available:
        input_ids = input_ids.cuda(device)
    
    with torch.no_grad():
        output = model(input_ids=input_ids)
        greedy = torch.max(output.logits, dim=-1).indices[0][len(prefix_token_id_list) - 1: -1]
        target = input_ids[0][len(prefix_token_id_list):]
        ratio = round(torch.mean(torch.eq(greedy, target).float()).item(), 4)
        return ratio

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoding_method", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--model_name", type=str)
    return parser.parse_args()

def get_config(decoding_method):
    prefix_len, decoding_len, number_of_instance_to_generate_per_method = 0, 0, 0

    if decoding_method == 'greedy':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 1
    elif decoding_method == 'beam':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 1
    elif decoding_method == 'topk':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 3
    elif decoding_method == 'nucleus':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 3
    elif decoding_method == 'contrastive':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 1
    elif decoding_method == 'resistance':
        prefix_len = 40
        decoding_len = 200
        number_of_instance_to_generate_per_method = 1
    
    return prefix_len, decoding_len, number_of_instance_to_generate_per_method

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda')

    args = parse_config()
    
    save_path = f"../inference_results/{args.model_name}/{args.data_name}/{args.decoding_method}/{args.decoding_method}_greedy_ratio_result.json".format(args.decoding_method)
    test_path = f"../inference_results/{args.model_name}/{args.data_name}/{args.decoding_method}/{args.decoding_method}_result.json"
    print ('evaluation save name is {}'.format(save_path))

    print ('Model loading...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded')

    print ('Data loading...')
    prefix_len, decoding_len, number_of_instance_to_generate_per_method = get_config(args.decoding_method)
    data = Data(tokenizer, test_path, args.data_path, prefix_len, decoding_len, number_of_instance_to_generate_per_method)
    print ('Data loaded')

    print ('Computing greedy ratio...')
    greedy_ratio = 0.0
    with torch.no_grad():
        greddy_same_count = 0
        for idx in tqdm(range(len(data.prefix_token_id_list))):
            for i in range(number_of_instance_to_generate_per_method):
                greedy_ratio += inference_one_instance(args, data.prefix_token_id_list[idx], data.result_token_id_list[idx][i], model, cuda_available, device)
        
        greedy_ratio = greedy_ratio / (len(data.prefix_token_id_list) * number_of_instance_to_generate_per_method)
        print ('Greedy ratio calculated completed')
        print ('greedy ratio: ', greedy_ratio)
    
    import json
    with open(save_path, 'w') as outfile:
        json.dump({'greedy_ratio': greedy_ratio, 'test_cases': len(data.prefix_token_id_list) * number_of_instance_to_generate_per_method}, outfile, indent=4)

    

