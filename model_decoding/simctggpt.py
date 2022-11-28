import sys
from tqdm import tqdm
import numpy as np
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel

from .utlis_ngram import * 
from .utlisgpt import ContrastiveDecodingOneStepFast

val_fct = CrossEntropyLoss(reduction='none')
class SimCTGGPT(nn.Module):
    def __init__(self, model_name, special_token_list=[]):
        super(SimCTGGPT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        if len(special_token_list) > 0:
            print ('Original vocabulary size is {}'.format(len(self.tokenizer)))
            print ('Adding special tokens...')
            self.tokenizer.add_tokens(special_token_list)
            print ('Special token added.')
            print ('Resizing language model embeddings...')
            self.model.resize_token_embeddings(len(self.tokenizer))
            print ('Language model embeddings resized.')
        self.vocab_size = len(self.tokenizer)
        print ('The vocabulary size of the language model is {}'.format(len(self.tokenizer)))
        self.embed_dim = self.model.config.hidden_size

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        return last_hidden_states, logits

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    @torch.no_grad()
    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len, 
        end_of_sequence_token_id = None, early_stop = False):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
           end_of_sequence_token_id: the token id that denotes the end of generation
           early_stop: whether to use the end_of_sequence_token_id to truncate the output
        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        self.model.eval()
        from .utlisgpt import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        batch_size, seqlen = input_ids.size()
        prefix_len = seqlen
        #generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)

        output = generated[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len,
        end_of_sequence_token_id = None, early_stop = False):
        '''
            sample_step: 
                number of steps to decode with nucleus sampling, 
                for the remaining steps we use contrastive search
            decoding_len: 
                the total number of generated tokens
            beam_width: 
                size of candidate pool during decoding
            alpha: 
                regulates importance of model confidence and degeneration penalty

        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        contrastive_step = decoding_len - sample_step
        _, prefix_len = input_ids.size()
        # first do sample
        input_ids = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+sample_step, 
                            top_p=nucleus_p,
                            top_k=0)
        # then do contrastive search
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, contrastive_step)
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def greedy_search(self, input_ids, decoding_len, end_of_sequence_token_id = None, early_stop = False, speedup=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        use_cache = True if speedup else False
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, use_cache=use_cache)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def beam_search(self, input_ids, beam_width, decoding_len, end_of_sequence_token_id = None, early_stop = False):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len, end_of_sequence_token_id = None, early_stop = False, speedup=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        use_cache = True if speedup else False
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0,
                            use_cache=use_cache)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def topk_sampling(self, input_ids, topk, decoding_len, end_of_sequence_token_id = None, early_stop = False, speedup=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        use_cache = True if speedup else False
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=1.0,
                            top_k=topk,
                            use_cache=use_cache)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    @torch.no_grad()
    def resistance_decoding(self, input_ids, beam_width, decoding_len, alpha=0.2,
        end_of_sequence_token_id = None, early_stop = False, resistance_function=None):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
           end_of_sequence_token_id: the token id that denotes the end of generation
           early_stop: whether to use the end_of_sequence_token_id to truncate the output
        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        if resistance_function == 'ours':
            ngram_lookup_table = ngram_lookup_table_ours
        elif resistance_function == 'constant':
            ngram_lookup_table = ngram_lookup_table_constant
        elif resistance_function == 'log':
            ngram_lookup_table = ngram_lookup_table_log
        elif resistance_function == 'exp':
            ngram_lookup_table = ngram_lookup_table_exp
        else:
            raise Exception(f"[!] Unknown resistance function: {resistance_function}")


        self.model.eval()
        batch_size, seqlen = input_ids.size()
        prefix_len = seqlen
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None

        # make the direct graph with a dict of dict
        graph = {}
        running_label = []
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, graph, label = ResistanceDecodingNGram(
                self.model,
                alpha,
                input_ids,
                beam_width,
                past_key_values,
                last_hidden_states,
                first_step=step == 0,
                graph=graph,
                token_list=generated[0],
                max_length=5,
                ngram_lookup_table=ngram_lookup_table
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
            if generated[0][-1] == end_of_sequence_token_id:
                break
        output = generated[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def detailed_greedy_search(self, input_ids, decoding_len, end_of_sequence_token_id = None, early_stop = False, speedup=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        generated = []
        ids = input_ids.clone()
        graph = {}

        # init the directed graph
        token_list = input_ids[0].tolist()
        for i in range(len(token_list)):
            node = token_list[i]
            if node not in graph:
                graph[node] = {
                    'next_neighbors': set(),
                    'indexes': [i]
                }
            if i > 0:
                last_node = token_list[i-1]
                graph[last_node]['next_neighbors'].add(node)
        last_node = token_list[-1]
        activate = False
        activate_results = []
        past_key_values = None
        for _ in range(decoding_len):
            output = self.model(input_ids=ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = output['past_key_values']
            next_token_logits = output['logits'][-1, -1, :]
            next_token_prob, next_token = next_token_logits.max(dim=-1)
            if next_token.item() == end_of_sequence_token_id:
                break

            if next_token.item() in graph and len(graph[next_token.item()]['indexes']) == 1:
                activate = True
                activate_results.append(next_token_prob.item())
            else:
                activate = False
                activate_results = []
            if activate and next_token.item() in graph and len(graph[next_token.item()]['indexes']) == 2 and len(activate_results) > 0:
                # degeneration are make
                ipdb.set_trace()
                break

            generated.append((next_token.item(), next_token_prob.item()))
            if next_token.item() not in graph:
                graph[next_token.item()] = {
                    'next_neighbors' : set(),
                    'indexes': [len(token_list)]
                }
            else:
                graph[next_token.item()]['indexes'].append(len(token_list))
            graph[last_node]['next_neighbors'].add(next_token.item())
            last_node = next_token.item()
            ids = next_token.view(1, 1)
        return [a for a, b in generated], np.mean(activate_results)


