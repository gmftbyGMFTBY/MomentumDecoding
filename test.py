from models import SimCTGGPT
import ipdb

# model = SimCTGGPT('uer/gpt2-chinese-cluecorpussmall')
model = SimCTGGPT('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
model.eval().cuda()
tokenizer = model.tokenizer
eos_token_id = tokenizer.eos_token_id
# eos_token_id = tokenizer.sep_token_id

while True:
    prefix = input('Prefix >>> ')
    input_ids = tokenizer(prefix, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
    output, _ = model.resistance_decoding(
        input_ids=input_ids, 
        beam_width=5, 
        alpha=0.2, 
        decoding_len=512, 
        end_of_sequence_token_id=eos_token_id, 
        early_stop=True, 
        resistance_function='ours'
    )
    # response = ''.join(tokenizer.convert_ids_to_tokens(output))
    response = tokenizer.decode(output)
    print('[Momentum Decoding]', response)
    print('=' * 50)

    # as a compraison, show the contrasive result
    output = model.fast_contrastive_search(
        input_ids=input_ids,
        beam_width=5,
        alpha=0.6,
        decoding_len=512,
        end_of_sequence_token_id=eos_token_id,
        early_stop=True
    )
    # response = ''.join(tokenizer.convert_ids_to_tokens(output))
    response = tokenizer.decode(output)
    print('[Contrastive Search]', response)
    print('=' * 50)
