import random
import copy

class cyclic_player(object):
    def __init__(self,offset_pos,
                      offset_career,
                      offset_vocb,
                      offset_special_token,
                      offset_relative_pos,
                      num_player):
        self.offset_pos = offset_pos
        self.offset_career = offset_career
        self.offset_vocb = offset_vocb
        self.offset_special_token = offset_special_token
        self.offset_relative_pos = offset_relative_pos

        self.num_player = num_player
        self.player_pos = [str(i) for i in range(self.num_player)]

    def __call__(self,one_action):

        #offset rand
        rand_offset = random.randint(0,self.num_player-1)
        shuffle_map = [(player_id + rand_offset) % self.num_player for player_id in range(self.num_player)]

        #replace system keys
        sys1_list = one_action["sys1"].split(" ")
        for token_idx in range(len(sys1_list)):
            if sys1_list[token_idx] in self.player_pos:
                sys1_list[token_idx] = str(shuffle_map[int(sys1_list[token_idx])])
        sys1_aug = " ".join(sys1_list)

        if one_action["sys2"]!=None:
            sys2_list = one_action["sys2"].split(" ")
            for token_idx in range(len(sys2_list)):
                if sys2_list[token_idx] in self.player_pos:
                    sys2_list[token_idx] = str(shuffle_map[int(sys2_list[token_idx])])
            sys2_aug = " ".join(sys2_list)
        else:
            sys2_aug = None

        #replace action ids only in act not in statement
        if one_action["act_type"]=="act":
            act_id_aug = shuffle_map[one_action["act_id"]]
        else:
            act_id_aug = one_action["act_id"]

        augmented_data = {"sys1": sys1_aug,
                          "sys2": sys2_aug,
                          "nlp1": one_action["nlp1"],
                          "nlp2": one_action["nlp2"],
                          "act_mask": one_action["act_mask"],
                          "act_type": one_action["act_type"],
                          "r_imme": one_action["r_imme"],
                          "r_finnal": one_action["r_finnal"],
                          "act_id": act_id_aug }
        return(augmented_data)