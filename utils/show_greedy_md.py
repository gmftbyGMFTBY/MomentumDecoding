from models import *
import ipdb

model = SimCTGGPT('gpt2-xl')
model.eval().cuda()
tokenizer = model.tokenizer
eos_token_id = tokenizer.eos_token_id

st = tokenizer.convert_ids_to_tokens(10000)[0]

input_ids = tokenizer('Slovakia eliminated defending champions Italy from the World Cup. First round groups E and F were decided on Thursday: Japan, Paraguay and the Netherlands progress alongside', return_tensors='pt')['input_ids'].cuda()
output, running_label = model.resistance_decoding(input_ids=input_ids, beam_width=5, alpha=0.2, decoding_len=256, end_of_sequence_token_id=eos_token_id, early_stop=True, resistance_function='ours')
running_label = [False] * 32 + running_label

assert len(output) == len(running_label)

output = tokenizer.convert_ids_to_tokens(output)
rest = []
for token, label in zip(output, running_label):
    rest.append(token + f'{label}')

string = ' '.join(rest)
print(string)
