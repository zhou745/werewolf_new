import numpy as np

class random_a_generator(object):

    def __init__(self,num_player,vocb_size):
        self.num_player = num_player
        self.vocb_size = vocb_size

    def __call__(self,head_token,nlp_state,action_mask,action_type):
        if action_type == "act":
            alive_player = []
            for player_id in range(len(action_mask)):
                if action_mask[player_id]>-0.5:
                    alive_player.append(player_id)

            a_policy = [1/len(alive_player) if player_id in alive_player else 0. for player_id in range(self.num_player)]
        elif action_type=="statement":
            a_policy = [1./self.vocb_size for i in  range(self.vocb_size)]
        else:
            raise RuntimeError("unknonw act type")

        return (np.array(a_policy,dtype=float))