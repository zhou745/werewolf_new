import torch
import numpy as np


class dict_timed_headtoken_generator_batch(object):

    def __init__(self,key_dict,offset_manager,device):
        self.key_dict = key_dict
        self.offset_manager = offset_manager
        self.device = device

    def __call__(self,sys_keys):
        #assert the head tokens

        head_token_ids = []
        for sys_key in sys_keys:
            current_key_t = []
            for key_t in sys_key:
                current_key_t.append(self.key_dict[key_t]+self.offset_manager)

            head_token_ids.append(current_key_t)

        head_token_ids_cuda = torch.tensor(head_token_ids,dtype=torch.int64).to(self.device)
        return (head_token_ids_cuda)