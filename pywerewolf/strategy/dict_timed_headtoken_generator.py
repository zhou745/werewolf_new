import torch
import numpy as np


class dict_timed_headtoken_generator(object):

    def __init__(self,key_dict,offset_manager,device):
        self.key_dict = key_dict
        self.offset_manager = offset_manager
        self.device = device

    def __call__(self,sys_key):
        #assert the head tokens

        head_token_ids = []

        for key_t in sys_key:
            head_token_ids.append(self.key_dict[key_t]+self.offset_manager)

        head_token_ids_cuda = torch.tensor(head_token_ids,dtype=torch.int64).to(self.device)
        return (head_token_ids_cuda)