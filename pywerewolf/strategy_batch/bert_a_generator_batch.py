import random
import torch
import numpy as np
import torch.nn.functional as F

class bert_a_generator_batch(object):

    def __init__(self,deep_model,tokenizer,device,aligned_size):
        self.deep_model = deep_model
        self.tokenizer = tokenizer
        self.device = device
        self.aligned_size = aligned_size

    def __call__(self,head_token_b,nlp_state_b,action_mask_b,action_type_b):
        if len(head_token_b)==0:
            return ({
            "act":[],
            "statement":[]
            })

        nlp_state_list = []
        attention_mask = []
        for nlp_state in nlp_state_b:
            one_nlp_state_ids = self.tokenizer(nlp_state)
            one_attention_mask = [1 for i in range(len(one_nlp_state_ids))]

            aligned_part = [0 for i in range(self.aligned_size-len(one_nlp_state_ids))]

            one_nlp_state_ids += aligned_part
            one_attention_mask += aligned_part
            nlp_state_list.append(one_nlp_state_ids)
            attention_mask.append(one_attention_mask)

        head_token_cuda = head_token_b
        nlp_state_cuda = torch.tensor(nlp_state_list,dtype=torch.int64).to(self.device)
        attention_mask_cuda = torch.tensor(attention_mask,dtype=torch.int64).to(self.device)
        act_type_cuda = torch.tensor(action_type_b=="act",dtype=torch.bool).to(self.device)

        action_mask_cuda = torch.tensor(action_mask_b,dtype=torch.float32).to(self.device)

        #reshape to 2d
        logit_value_cuda = self.deep_model(head_token_cuda,nlp_state_cuda,act_type_cuda,attention_mask = attention_mask_cuda)

        if logit_value_cuda["act"]!=None:
            logit_value_cuda["act"] += action_mask_cuda[act_type_cuda]

        a_policy_dict = {
            "act":[],
            "statement":[]
        }
        if logit_value_cuda["act"]!=None:
            a_policy_dict["act"] = F.softmax(logit_value_cuda["act"],dim=-1).detach().cpu().numpy()

        if logit_value_cuda["statement"]!=None:
            a_policy_dict["statement"] = F.softmax(logit_value_cuda["statement"],dim=-1).detach().cpu().numpy()
        return (a_policy_dict)