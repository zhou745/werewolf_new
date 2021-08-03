import torch
import numpy as np


class bert_headtoken_generator_batch(object):

    def __init__(self,deep_model,tokenizer,device,aligned_size = 64):
        self.deep_model = deep_model
        self.tokenizer = tokenizer
        self.device = device
        self.aligned_size = aligned_size

    def __call__(self,head_token_b):
        #assert the head tokens

        head_token_ids = []
        attention_mask = []
        for headtoken in head_token_b:
            one_headtoken_ids = self.tokenizer(headtoken)
            one_attention_mask = [1 for i in range(len(one_headtoken_ids))]

            aligned_part = [0 for i in range(self.aligned_size-len(one_headtoken_ids))]

            one_headtoken_ids += aligned_part
            one_attention_mask += aligned_part
            head_token_ids.append(one_headtoken_ids)
            attention_mask.append(one_attention_mask)

        head_token_cuda = torch.tensor(head_token_ids,dtype=torch.int64).to(self.device)
        attention_mask_cuda = torch.tensor(attention_mask,dtype=torch.int64).to(self.device)

        head_token_vector = self.deep_model(head_token_cuda,attention_mask = attention_mask_cuda)

        return (head_token_vector)