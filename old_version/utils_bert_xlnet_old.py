#!/usr/bin/env python3

'''
@Time   : 2020-01-14 16:28:27
@Author : su.zhu
@Desc   : 
'''

import itertools

import torch

from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel

MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer),
        'xlnet': (XLNetModel, XLNetTokenizer),
        }

def load_pretrained_transformer(model_type, model_name, lowercase=False):
	pretrained_model_class, tokenizer_class = MODEL_CLASSES[model_type]
	tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=lowercase)
	#pretrained_model = pretrained_model_class.from_pretrained(model_name, output_hidden_states = True)
	pretrained_model = pretrained_model_class.from_pretrained(model_name)
	print(pretrained_model.config)
	return tokenizer, pretrained_model

def prepare_inputs_for_bert_xlnet(sorted_examples, tokenizer, bos_eos=False, cls_token_at_end=False, pad_on_left=False, pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None, feed_transformer=False):
    """
    TODO: if feed_transformer == True, select CLS output as the first embedding of cls_token
    """
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,        # input_ids
        'segments': segments_tensor,    # token_type_ids
        'mask': input_mask,             # attention_mask
        'selects': selects_tensor,      # original_word_to_token_position
        'copies': copies_tensor         # original_word_position
        }
    """
    ## sentences are sorted by sentence length
    cls_token = tokenizer.cls_token # [CLS]
    sep_token = tokenizer.sep_token # [SEP]
    if bos_eos:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
    
    word_lengths = []
    tokens = []
    segment_ids = []
    selected_indexes = []
    start_pos = 0
    for example_index, example in enumerate(sorted_examples):
        words = example
        if bos_eos:
            words = [bos_token] + words + [eos_token]
        word_lengths.append(len(words))
        selected_index = []
        ts = []
        for w in words:
            if cls_token_at_end:
                selected_index.append(len(ts))
            else:
                selected_index.append(len(ts) + 1)
            ts += tokenizer.tokenize(w)
        ts += [sep_token]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        #print(ts)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
    max_length_of_tokens = max([len(tokenized_text) for tokenized_text in tokens])
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
        selected_indexes = [[padding_lengths[idx] + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
        selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    max_length_of_sentences = max(word_lengths) # the length is already +2 when bos_eos is True.
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(length)] for idx,length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)), dtype=torch.long, device=device)
    #return {'tokens': tokens_tensor, 'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}
    return tokens_tensor, segments_tensor, input_mask, selects_tensor, copies_tensor

def transformer_forward_by_ignoring_suffix(transformer, batch_size, max_word_length, input_ids, segment_ids, selects, copies, attention_mask, device=None):
    '''
    Ignore hidden states of all suffixes: [CLS] from ... to de ##n ##ver [SEP] => from ... to de
    '''
    outputs = transformer(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
    pretrained_top_hiddens = outputs[0]
    #print(pretrained_top_hiddens[-1])
    batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
    chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
    embeds = torch.zeros(batch_size * max_word_length, hidden_size, device=device)
    embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(batch_size, max_word_length, -1)
    #print(embeds[-1])
    return embeds

