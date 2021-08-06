import copy

from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base

class werewolf_manager_timed(werewolf_manager_base):
    def __init__(self,num_player,game_compose=None,
                                 game_step=None,
                                 player_size_reserve=None,
                                 player_career_reserve=None,
                                 player_vocb_reserve=None,
                                 special_token_reserve=None,
                                 max_time = 12):

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

        super(werewolf_manager_timed, self).__init__(num_player,**hyper_para)

        #in timed manager, system state is returned by a list
        self.max_time = max_time
        self.last_time_len = [0 for i in range(self.num_player)]
        self.current_time_idx = 0
        self.player_system_timed_state = [["" for time in range(self.max_time)] for i in range(self.num_player)]


    def progress_time(self):
        werewolf_manager_base.progress_time(self)
        for player_id in range(self.num_player):
            self.last_time_len[player_id] = len(self.player_system_state[player_id])
        self.current_time_idx +=1

    def reset(self,career_assign=None):
        werewolf_manager_base.reset(self,career_assign=career_assign)

        self.last_time_len = [0 for i in range(self.num_player)]
        self.current_time_idx = 0
        self.player_system_timed_state = [["" for time in range(self.max_time)] for i in range(self.num_player)]

    def step(self,action_list):
        assert self.ready==True
        #get the function to progress the game
        action_fun = eval("self."+self.game_step[self.current_progress])
        #progress the game
        self.progress_idx()
        #get the game evolution
        system_state, nlp_state, action_mask,action_type,reward,status,current_time = action_fun(action_list)

        for player_id in range(self.num_player):
            self.player_system_timed_state[player_id][self.current_time_idx] = \
                copy.copy(self.player_system_state[player_id][self.last_time_len[player_id]:])

        return(copy.deepcopy(self.player_system_timed_state),
               copy.copy(nlp_state),
               copy.copy(action_mask),
               copy.copy(action_type),
               copy.copy(reward),
               copy.copy(status),
               copy.copy(current_time))