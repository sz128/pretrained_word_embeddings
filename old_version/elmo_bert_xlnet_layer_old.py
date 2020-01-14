#!/usr/bin/env python3

'''
@Time   : 2019-11-29 16:08:49
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids
from utils_bert_xlnet_old import *

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

    sentences = ['hello world', 'may i help you ?', 'i dont care', 'wifi']
    words = [s.split(' ') for s in sentences]
    
    print(bert_model(words))
    print(elmo_model(words))
