import torch


class bert_headtoken_generator(object):

    def __init__(self,deep_model,tokenizer,device):
        self.deep_model = deep_model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self,head_token):
        #convert string to cuda tensor
        assert isinstance(head_token,str)

        head_token_cuda = torch.tensor(self.tokenizer(head_token),dtype=torch.int64).to(self.device)

        #reshape to 2d
        head_token_cuda = head_token_cuda.view(1,-1)

        head_token_vector = self.deep_model(head_token_cuda)

        return (head_token_vector)