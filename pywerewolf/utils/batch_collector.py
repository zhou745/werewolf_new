class batch_collector(object):
    def __init__(self,tokenizer,aligned_size,num_player):
        self.tokenizer = tokenizer
        self.aligned_size = aligned_size
        self.num_player = num_player

        self.mask_none = [0 for i in range(self.num_player)]

    def __call__(self,sampled_data):

        s1_list = []
        s1_atten_musk_list = []
        nlp1_list = []
        nlp1_atten_musk_list = []

        action_musk = []
        action_type = []

        action_ids = []
        finnal_reward = []

        for one_action in sampled_data:
            s1_tokens = self.tokenizer(one_action["sys1"])
            nlp1_tokens = self.tokenizer(one_action["nlp1"])

            s1_atten_musk = [1 for i in range(len(s1_tokens))]
            nlp1_atten_musk = [1 for i in range(len(nlp1_tokens))]

            align_s1 = [0 for i in range(self.aligned_size-len(s1_tokens))]
            align_nlp1 = [0 for i in range(self.aligned_size-len(nlp1_tokens))]

            #align s1 nlp1
            s1_tokens += align_s1
            nlp1_tokens += align_nlp1

            s1_atten_musk += align_s1
            nlp1_atten_musk += align_nlp1


            s1_list.append(s1_tokens)
            nlp1_list.append(nlp1_tokens)
            s1_atten_musk_list.append(s1_atten_musk)
            nlp1_atten_musk_list.append(nlp1_atten_musk)

            act_musk = one_action["act_mask"] if one_action["act_mask"]!=None else self.mask_none
            action_musk.append(act_musk)
            action_type.append(one_action["act_type"]=="act")
            finnal_reward.append(one_action["r_finnal"])
            action_ids.append(one_action["act_id"])

        return ({
            "s1":s1_list,
            "s1_atten_mask":s1_atten_musk_list,
            "nlp1":nlp1_list,
            "nlp1_atten_mask":nlp1_atten_musk_list,
            "act_mask":action_musk,
            "act_ids":action_ids,
            "act_type":action_type,
            "finnal_reward":finnal_reward
        })

class batch_timed_collector(object):
    def __init__(self,tokenizer,aligned_size,num_player,key_dict,offset_game):
        self.tokenizer = tokenizer
        self.aligned_size = aligned_size
        self.num_player = num_player
        self.key_dict = key_dict
        self.offset_game = offset_game

        self.mask_none = [0 for i in range(self.num_player)]

    def __call__(self,sampled_data):

        s1_list = []
        s1_atten_musk_list = []
        nlp1_list = []
        nlp1_atten_musk_list = []

        action_musk = []
        action_type = []

        action_ids = []
        finnal_reward = []

        for one_action in sampled_data:
            s1_tokens = []
            for key in one_action["sys1"]:
                s1_tokens.append(self.key_dict[key]+self.offset_game)
            nlp1_tokens = self.tokenizer(one_action["nlp1"])

            s1_atten_musk = [1 for i in range(len(s1_tokens))]
            nlp1_atten_musk = [1 for i in range(len(nlp1_tokens))]
            align_nlp1 = [0 for i in range(self.aligned_size-len(nlp1_tokens))]

            #align s1 nlp1
            nlp1_tokens += align_nlp1
            nlp1_atten_musk += align_nlp1


            s1_list.append(s1_tokens)
            nlp1_list.append(nlp1_tokens)
            s1_atten_musk_list.append(s1_atten_musk)
            nlp1_atten_musk_list.append(nlp1_atten_musk)

            act_musk = one_action["act_mask"] if one_action["act_mask"]!=None else self.mask_none
            action_musk.append(act_musk)
            action_type.append(one_action["act_type"]=="act")
            finnal_reward.append(one_action["r_finnal"])
            action_ids.append(one_action["act_id"])

        return ({
            "s1":s1_list,
            "s1_atten_mask":s1_atten_musk_list,
            "nlp1":nlp1_list,
            "nlp1_atten_mask":nlp1_atten_musk_list,
            "act_mask":action_musk,
            "act_ids":action_ids,
            "act_type":action_type,
            "finnal_reward":finnal_reward
        })

