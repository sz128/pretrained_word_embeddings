#!/usr/bin/env python3

'''
@Time   : 2019-11-29 16:08:49
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids
from utils_bert_xlnet import *

class PretrainedInputEmbeddings(nn.Module):
    """Construct the embeddings from pre-trained ELMo\BERT\XLNet embeddings
    1) pretrained_model_type == 'tf', pretrained_model_info = {'type': 'bert', 'name': 'bert-base-uncased', 'alignment': 'first'}
        1.1): bert, bert-base-uncased, bert-base-cased, bert-base-chinese
        1.2): xlnet, xlnet-base-cased
        1.3): alignemnt can be None, 'first' and 'avg'
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
            if 'alignment' not in self.pretrained_model_info or self.pretrained_model_info['alignment'] not in {'first', 'avg', 'ori'}:
                self.alignment = None
            else:
                self.alignment = self.pretrained_model_info['alignment']
            self.embedding_dim = self.tf_model.config.hidden_size
        else:
            self.elmo_model = Elmo(pretrained_model_info['elmo_json'], pretrained_model_info['elmo_weight'], 1, dropout=0)
            self.embedding_dim = self.elmo_model.get_output_dim()

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, words, no_dropout=False):
        """
        words: a list of word list
        """
        """
        [NOTE]: If you want to feed output word embeddings into RNN/GRU/LSTM by using pack_padded_sequence, you'd better sort 'words' by length in advance.
        """
        if self.pretrained_model_type == 'tf':
            input_tf, tf_tokens, output_tokens, output_token_lengths = prepare_inputs_for_bert_xlnet(words, self.tf_tokenizer, device=self.device, **self.tf_input_args, alignment=self.alignment)
            embeds = transformer_forward_by_ignoring_suffix(self.tf_model, **input_tf, device=self.device, alignment=self.alignment)
        else:
            tokens = batch_to_ids(words).to(self.device)
            elmo_embeds = self.elmo_model(tokens)
            embeds = elmo_embeds['elmo_representations'][0]
            output_tokens = words
            output_token_lengths = [len(ws) for ws in words]
        
        if not no_dropout:
            embeds = self.dropout_layer(embeds)

        return embeds, output_tokens, output_token_lengths


if __name__ == "__main__":
    bert_model_ori = PretrainedInputEmbeddings(pretrained_model_type='tf', pretrained_model_info={'type': 'bert', 'name': 'bert-base-uncased', 'alignment': 'ori'}, dropout=0.0, device=None)
    bert_model_first = PretrainedInputEmbeddings(pretrained_model_type='tf', pretrained_model_info={'type': 'bert', 'name': 'bert-base-uncased', 'alignment': 'first'}, dropout=0.0, device=None)
    bert_model_avg = PretrainedInputEmbeddings(pretrained_model_type='tf', pretrained_model_info={'type': 'bert', 'name': 'bert-base-uncased', 'alignment': 'avg'}, dropout=0.0, device=None)
    elmo_model = PretrainedInputEmbeddings(pretrained_model_type='elmo', pretrained_model_info={'elmo_json': 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', 'elmo_weight': 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'}, dropout=0.0, device=None)

    sentences = ['hello world', 'may i help you ?', 'i dont care', 'wifi']
    words = [s.split(' ') for s in sentences]
    
    print(bert_model_ori(words))
    print(bert_model_first(words))
    print(bert_model_avg(words))
    print(elmo_model(words))
