from transformers import BertModel,BertConfig
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_size,output_size,intermedia=1024):
        super().__init__()
        # self.layernorm_1 = torch.nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size,intermedia)
        self.layernorm_1 = torch.nn.LayerNorm(intermedia)
        self.fc2 = nn.Linear(intermedia,output_size)

        for m in self.children():
            if isinstance(m,torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            else:
                # torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.uniform_(m.weight,-0.001,0.001)
                torch.nn.init.zeros_(m.bias)

    def forward(self,x):
        # l1 = self.layernorm_1(x)
        h1 = F.relu(self.fc1(x))
        l1 = self.layernorm_1(h1)
        # l2 = self.layernorm_2(h1)
        h2 = self.fc2(l1)
        return(h2)

class headtoken_model(nn.Module):
    def __init__(self,config,mlp_intermedia = 1024):
        super().__init__()

        self.mlp_intermedia =mlp_intermedia
        self.bert_backbone = BertModel(config)
        self.headtoken_mlp = MLP(config.hidden_size,config.hidden_size,intermedia=self.mlp_intermedia)


    def forward(self,head_token,attention_mask=None):

        if attention_mask==None:
            h1 = self.bert_backbone(head_token)
        else:
            h1 = self.bert_backbone(head_token,attention_mask=attention_mask)

        h_out = h1[0][:, 0, :] + 0. * h1[0][:, :, :].mean(axis=1) + h1[1][:, :] * 0.
        return(h_out)