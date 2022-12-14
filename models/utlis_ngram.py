import sys
import ipdb
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import random
from math import *


def ngram_lookup_table_ours(rep_len, mapping=None):
    if mapping is None:
        # for GPT2-XL
        mapping = {
            0: 0.0,
            1: 1.0,    # no rersistance
            2: 3.0,
            3: 4.0,
            4: 5.0,
            5: 5.0,
        }
        # for IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
        # mapping = {
        #     0: 0.0,
        #     1: 1.0,    # no rersistance
        #     # 2: 3.0,
        #     2: 1.0,
        #     3: 2.0,
        #     4: 3.0,
        #     5: 4.0,
        # }
    if rep_len in mapping:
        return mapping[rep_len]
    else:
        return np.inf

def ResistanceDecodingNGram(
    model, 
    scale_alpha,
    ids, 
    beam_width, 
    past_key_values,
    last_hidden_states,
    first_step=False,
    graph={},
    token_list=None,
    max_length=5,
    opt=False,
    seqlen=-1,
    ngram_lookup_table=ngram_lookup_table_ours,
    ):
    if opt:
        if first_step:
            attention_mask = torch.ones_like(ids)
        else:
            attention_mask = torch.ones((1, seqlen)).to(ids.device)
        output = model(
            input_ids=ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
    else:
        output = model(
            input_ids=ids, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

    past_key_values = output.past_key_values
    last_hidden_states = output.hidden_states[-1]    # [B, S, E]
    logit = output.logits[:, -1]
    if first_step:
        token_list = ids[0].tolist()
        for i in range(len(token_list)):
            node = token_list[i]
            if node not in graph:
                graph[node] = {
                    'next_neighbors': set(),
                    'indexes': [i]
                }
            else:
                graph[node]['indexes'].append(i)
            if i > 0:
                last_node = token_list[i-1]
                graph[last_node]['next_neighbors'].add(node)
        last_node = token_list[-1]
    else:
        last_node = ids[0].item()

    bsz, seqlen, embed_dim = last_hidden_states.size()
    next_probs = F.softmax(logit, dim=-1)
    _, top_k_ids = torch.topk(logit, dim=-1, k=beam_width)
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)

    rep_lens = [find_max_repetition_length(graph, token_list, i, max_length=max_length) for i in top_k_ids.tolist()[0]]
    alphas = [ngram_lookup_table(i) for i in rep_lens]
    scores = []
    for p, alpha in zip(top_k_probs[0], alphas):
        s = p - scale_alpha * alpha
        scores.append(s)
    scores = torch.stack(scores)
    _, index = scores.max(dim=-1)
    next_id = top_k_ids[0, index].reshape(1, 1)

    # if index == 0:
    #     is_greedy = True
    # else:
    #     is_greedy = False
    if top_k_ids[0, 0].item() not in graph:
        is_greedy = 0
    else:
        if index == 0:
            is_greedy = 1
        else:
            is_greedy = 2

    # update the graph
    if next_id.item() not in graph:
        graph[next_id.item()] = {
            'next_neighbors': set() ,
            'indexes': [len(token_list)]
        }
    else:
        graph[next_id.item()]['indexes'].append(len(token_list))
    graph[last_node]['next_neighbors'].add(next_id.item())
    return next_id, past_key_values, last_hidden_states, graph, is_greedy

def ngram_lookup_table_log(rep_len, mapping=None):
    if rep_len == 0:
        return 0
    # avoid alpha=0.2
    return log(1+rep_len, 6)/0.2

def ngram_lookup_table_exp(rep_len, mapping=None):
    if rep_len == 0:
        return 0
    # avoid alpha=0.2
    return 1.5**rep_len/2

def ngram_lookup_table_constant(rep_len, mapping=None):
    if mapping is None:
        mapping = {
            0: 0.0,
            1: 2.0,    # no rersistance
            2: 2.0,
            3: 2.0,
            4: 2.0,
            5: 2.0,
        }
    if rep_len in mapping:
        return mapping[rep_len]
    else:
        return 2


def find_max_repetition_length(graph, token_list, candidate, max_length=5):
    '''candidate must in the graph; generate it will lead to at least a circuit;
    we aim to find all the potential circuits and count its repetition length
    to set the hyper-parameter for resistance function term'''
    if candidate not in graph:
        return 0
    indexes = graph[candidate]['indexes']
    index_path = {i: set() for i in range(1, max_length+1)}

    for index in indexes:
        idx = index
        while idx >= 0:
            item = tuple(token_list[idx:index+1])
            idx -= 1
            if len(item) <= max_length:
               index_path[len(item)].add(item)
            else:
                break
    l = 1
    for i in range(max_length, 1, -1):
        query = tuple(token_list[-i+1:] + [candidate])
        if query in index_path[i]:
            l = i
            break
    return l
