import random
import torch
import numpy as np

class bert_q_generator(object):

    def __init__(self,deep_model,tokenizer,device):
        self.deep_model = deep_model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self,head_token,nlp_state,action_mask,action_type):
        #convert string to cuda tensor
        if isinstance(head_token,int):
            head_token_cuda = torch.tensor(head_token,dtype=torch.int64).to(self.device)
        elif isinstance(head_token,torch.Tensor):
            head_token_cuda = head_token
        else:
            raise RuntimeError("unknonw head token type")

        nlp_state_cuda = torch.tensor(self.tokenizer(nlp_state),dtype=torch.int64).to(self.device)
        act_type_cuda = torch.tensor([action_type=="act"],dtype=torch.bool).to(self.device)

        #reshape to 2d
        head_token_cuda = head_token_cuda.view(1,-1)
        nlp_state_cuda = nlp_state_cuda.view(1,-1)

        q_value_cuda = self.deep_model(head_token_cuda,nlp_state_cuda,act_type_cuda)[action_type]
        q_value = q_value_cuda.cpu().detach().numpy()[0]

        if action_mask!=None:
            q_value += np.array(action_mask)

        return (q_value)