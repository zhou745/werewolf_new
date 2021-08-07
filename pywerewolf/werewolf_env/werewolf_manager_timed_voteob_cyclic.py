from pywerewolf.werewolf_env.werewolf_manager_timed_cyclic import werewolf_manager_timed_cyclic
import numpy as np
import random

class werewolf_manager_timed_voteob_cyclic(werewolf_manager_timed_cyclic):
    def __init__(self,num_player,game_compose=None,
                                 game_step=None,
                                 player_size_reserve=None,
                                 player_career_reserve=None,
                                 player_vocb_reserve=None,
                                 special_token_reserve=None,
                                 relative_pos_reserve=100,
                                 max_time = 12):

        hyper_para = {"max_time":max_time}
        if game_step!=None:
            hyper_para.update({"game_step":game_step})
        if game_compose!=None:
            hyper_para.update({"game_compose": game_compose})
        if player_size_reserve!=None:
            hyper_para.update({"player_size_reserve": player_size_reserve})
        if player_career_reserve!=None:
            hyper_para.update({"player_career_reserve": player_career_reserve})
        if player_vocb_reserve!=None:
            hyper_para.update({"player_vocb_reserve": player_vocb_reserve})
        if special_token_reserve!=None:
            hyper_para.update({"special_token_reserve": special_token_reserve})

        super(werewolf_manager_timed_voteob_cyclic, self).__init__(num_player,**hyper_para)

    def vote_for_one(self, action):
        # !ToDo currently use majority voting
        vote_list = []
        for player_id in self.alive_list:
            vote_list.append(action[player_id])
            # !ToDo currently disable village leader extra vote
            # if idx == self.leader:
            #     vote_list.append(vote)
        # record the voting of each player
        vote_list_np = np.array(vote_list)
        state_vote = np.bincount(vote_list_np)
        voted_one = state_vote.argmax()
        samevote_list = []
        # duplicate check
        for idx in range(len(state_vote)):
            if state_vote[idx] == state_vote[voted_one]:
                samevote_list.append(idx)
        if len(samevote_list) > 1:
            voted = random.sample(samevote_list, 1)[0]
        else:
            voted = voted_one
        # tell the whole village who is voted out
        voted_out = "voted_out " + str(voted + self.offset_position)
        self.current_dead.append(voted)
        self.whole_village_progress(voted_out)
        info_text_list = []
        for player_id in self.alive_list:
            info_text_list.append("player_vote " + str(player_id + self.offset_position) +
                                  " " + str(action[player_id] + self.offset_position))
        info_text = " ".join(info_text_list)
        self.whole_village_info(info_text)

        # remove the dead player
        self.alive_list.remove(voted)
        self.type_id[self.card_to_player[voted]].remove(voted)

        # create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        # create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        # check terminal
        status = self.check_terminal()

        if status == "continue":
            reward = [0. for player_id in range(self.num_player)]
        elif status == "tied_game":
            reward = [0. for player_id in range(self.num_player)]
        elif status == "werewolf_win":
            reward = [1 if self.card_to_player[player_id] == 7 else -1
                      for player_id in range(self.num_player)]
        else:
            reward = [-1 if self.card_to_player[player_id] == 7 else 1
                      for player_id in range(self.num_player)]

        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)