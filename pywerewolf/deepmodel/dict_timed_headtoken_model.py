from transformers import BertModel,BertConfig
import torch
from torch import nn
import torch.nn.functional as F

BertConfig = BertConfig

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

class dict_timed_headtoken_model(nn.Module):
    def __init__(self,config,num_player,vocb_size,mlp_intermedia=1024,
                                                  max_time = 12):
        super().__init__()
        self.num_player = num_player
        self.vocb_size = vocb_size

        self.bert_backbone = BertModel(config)
        self.mlp_intermedia = mlp_intermedia
        self.max_time = max_time
        assert self.max_time>1
        self.hidden_size = config.hidden_size

        self.act_mlp = MLP(config.hidden_size*self.max_time,self.num_player,intermedia=self.mlp_intermedia)
        self.nlp_mlp  = MLP(config.hidden_size*self.max_time,self.vocb_size,intermedia=self.mlp_intermedia)


    def forward(self,head_token,nlp_state,act_type,attention_mask=None):
        bert_input = torch.cat((head_token,nlp_state),dim=1)
        if attention_mask==None:
            h1 = self.bert_backbone(bert_input)
        else:
            ones_begin = torch.ones_like(head_token).to(head_token.device)
            attention_mask_new = torch.cat((ones_begin,attention_mask),dim=1)
            h1 = self.bert_backbone(bert_input,attention_mask=attention_mask_new)

        #compute the mask first
        act_mask = act_type
        nlp_mask = ~act_mask

        result = {"act":None,
                  "statement":None}
        if act_mask.sum()>0:
            h_act = h1[0][act_mask,0:self.max_time,:]+0.*h1[0][act_mask,:,:].mean()+h1[1][act_mask,:].mean()*0.
            h_act = h_act.view(-1,self.max_time*self.hidden_size)
            logits_act = self.act_mlp(h_act)
            result["act"] = logits_act
        if nlp_mask.sum()>0:
            h_nlp = h1[0][nlp_mask,0:self.max_time,:]+0.*h1[0][nlp_mask,:,:].mean()+h1[1][nlp_mask,:].mean()*0.
            h_nlp = h_nlp.view(-1, self.max_time * self.hidden_size)
            logits_nlp = self.nlp_mlp(h_nlp)
            result["statement"] = logits_nlp
        return(result)