import random

class uniform_sampler(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size

    def __call__(self,sample_dict,sample_num):
        assert isinstance(sample_dict,list)

        sample_size = min(self.batch_size,sample_num)
        return (random.sample(sample_dict[0:sample_num],sample_size))
