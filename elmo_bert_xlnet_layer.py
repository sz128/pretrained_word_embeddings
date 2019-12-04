#!/usr/bin/env python3

'''
@Time   : 2019-11-29 16:08:49
@Author : su.zhu
@Desc   : 
'''

import itertools

import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids
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
    batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
    chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
    embeds = torch.zeros(batch_size * max_word_length, hidden_size, device=device)
    embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(batch_size, max_word_length, -1)
    return embeds

class PretrainedInputEmbeddings(nn.Module):
    """Construct the embeddings from pre-trained ELMo\BERT\XLNet embeddings
    1) pretrained_model_type == 'tf', pretrained_model_info = {'type': 'bert', 'name': 'bert-base-uncased'}
        1.1): bert, bert-base-uncased, bert-base-cased, bert-base-chinese
        1.2): xlnet, xlnet-base-cased
    2) pretrained_model_type == 'elmo', pretrained_model_info = {'elmo_json': '', 'elmo_weight': ''}
    """
    def __init__(self, pretrained_model_type='tf', pretrained_model_info={}, dropout=0.0, device=None):
        super(PretrainedInputEmbeddings, self).__init__()
        
        self.pretrained_model_type = pretrained_model_type.lower()
        self.pretrained_model_info = pretrained_model_info
        self.device = device
        
        assert self.pretrained_model_type in {'tf', 'elmo'}
        if self.pretrained_model_type == 'tf': 
            if 'uncased' in pretrained_model_info['name']:
                input_word_lowercase = True
            else:
                input_word_lowercase = False
            self.tf_tokenizer, self.tf_model = load_pretrained_transformer(self.pretrained_model_info['type'], self.pretrained_model_info['name'], lowercase=input_word_lowercase)
            #self.tf_model.embeddings.word_embeddings = nn.Embedding(6500, 768, padding_idx=0)
            #self.tf_model.encoder.layer = self.tf_model.encoder.layer[:2]
            self.tf_input_args = {
                    'cls_token_at_end': bool(self.pretrained_model_info['type'] in ['xlnet']),  # xlnet has a cls token at the end
                    'cls_token_segment_id': 2 if self.pretrained_model_info['type'] in ['xlnet'] else 0,
                    'pad_on_left': bool(self.pretrained_model_info['type'] in ['xlnet']), # pad on the left for xlnet
                    'pad_token_segment_id': 4 if self.pretrained_model_info['type'] in ['xlnet'] else 0,
                    }
            self.embedding_dim = self.tf_model.config.hidden_size
        else:
            self.elmo_model = Elmo(pretrained_model_info['elmo_json'], pretrained_model_info['elmo_weight'], 1, dropout=0)
            self.embedding_dim = self.elmo_model.get_output_dim()

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, words, no_dropout=False):
        """
        words: a list of word list
        """
        if self.pretrained_model_type == 'tf':
            lengths = [len(ws) for ws in words]
            input_tf_ids, tf_segment_ids, tf_attention_mask, tf_output_selects, tf_output_copies = prepare_inputs_for_bert_xlnet(words, self.tf_tokenizer, device=self.device, **self.tf_input_args)
            input_tf = {
                "input_ids": input_tf_ids,
                "segment_ids": tf_segment_ids,
                "attention_mask": tf_attention_mask,
                "selects": tf_output_selects,
                "copies": tf_output_copies,
                "batch_size": len(lengths),
                "max_word_length": max(lengths)
                }
            embeds = transformer_forward_by_ignoring_suffix(self.tf_model, **input_tf, device=self.device)
        else:
            tokens = batch_to_ids(words).to(self.device)
            elmo_embeds = self.elmo_model(tokens)
            embeds = elmo_embeds['elmo_representations'][0]
        
        if not no_dropout:
            embeds = self.dropout_layer(embeds)

        return embeds


if __name__ == "__main__":
    bert_model = PretrainedInputEmbeddings(pretrained_model_type='tf', pretrained_model_info={'type': 'bert', 'name': 'bert-base-uncased'}, dropout=0.0, device=None)
    elmo_model = PretrainedInputEmbeddings(pretrained_model_type='elmo', pretrained_model_info={'elmo_json': 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', 'elmo_weight': 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'}, dropout=0.0, device=None)

    sentences = ['hello world', 'may i help you ?', 'good']
    words = [s.split(' ') for s in sentences]
    
    print(bert_model(words))
    print(elmo_model(words))
