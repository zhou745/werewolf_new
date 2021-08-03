import random
import numpy as np

class random_q_generator(object):

    def __init__(self,num_player,vocb_size):
        self.num_player = num_player
        self.vocb_size = vocb_size

    def __call__(self,head_token,nlp_state,action_mask,action_type):
        if action_type == "act":
            while 1:
                idx_max = random.randint(0,self.num_player-1)
                if action_mask[idx_max]>-0.5:
                    break

            q_value = [0. if player_id != idx_max else 1 for player_id in range(self.num_player)]
        elif action_type=="statement":
            idx_max = random.randint(0, self.vocb_size)
            q_value = [0. if word_id != idx_max else 1 for word_id in range(self.vocb_size)]
        else:
            raise RuntimeError("unknonw act type")

        return (np.array(q_value,dtype=float))