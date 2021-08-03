from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base

class werewolf_manager_cyclic(werewolf_manager_base):
    def __init__(self,num_player,game_compose=None,
                                 game_step=None,
                                 player_size_reserve=None,
                                 player_career_reserve=None,
                                 player_vocb_reserve=None,
                                 special_token_reserve=None,
                                 relative_pos_reserve=100):

        hyper_para = {}
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

        super(werewolf_manager_cyclic, self).__init__(num_player,**hyper_para)

        #update the new offset index
        self.offset_relative_pos = self.special_token_reserve + self.offset_special_token
        self.offset_game_all = self.offset_relative_pos + relative_pos_reserve

        self.player_pos = [str(i+self.offset_position) for i in range(self.num_player)]

    def whole_village_info(self,text):
        language_list = text.split(" ")
        #use relative position in the language part
        for player_id in self.alive_list:
            current_language_list = []
            for token in language_list:
                if token in self.player_pos:
                    current_language_list.append(str((int(token)-(player_id+self.offset_position))%self.num_player
                                                     +self.offset_relative_pos))
                else:
                    current_language_list.append(token)
            self.player_nlp_state[player_id] +=" "+" ".join(current_language_list)