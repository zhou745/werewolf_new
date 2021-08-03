from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base
import random
import copy

class werewolf_manager_named(werewolf_manager_base):
    def __init__(self,num_player,num_names,game_compose=None,
                                            game_step=None,
                                             player_size_reserve=None,
                                             player_career_reserve=None,
                                             player_vocb_reserve=None,
                                             special_token_reserve=None,
                                             name_num_reserve=100):

        #player position player career player name
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

        self.num_names = num_names
        super(werewolf_manager_named, self).__init__(num_player,**hyper_para)

        #update the new offset index
        self.offset_name = self.special_token_reserve + self.offset_special_token
        self.offset_game_all =   self.offset_name + name_num_reserve

        self.player_pos = [str(i+self.offset_position) for i in range(self.num_player)]
        self.name_dict = [str(i+self.offset_name) for i in range(self.num_names)]

        #current names
        self.current_names = random.sample(self.name_dict,self.num_player)

    def reset(self, career_assign=None):
        # reset the game

        # reset the game step back to 0
        self.current_progress = 0
        self.current_time = "day 0"

        # reset game progress
        self.player_system_state = []
        self.player_nlp_state = []
        self.alive_list = []
        self.current_dead = []
        self.type_id = [[] for i in range(12)]

        # reset the game career
        if career_assign == None:
            random.shuffle(self.card_to_player)
        else:
            self.card_to_player = copy.copy(career_assign)

        # set the career type
        for i in range(self.num_player):
            # append player i to type self.card_to_player[i]
            self.type_id[self.card_to_player[i]].append(i)
            self.alive_list.append(i)

        #current names
        self.current_names = random.sample(self.name_dict,self.num_player)

        # initialize all the game state
        for i in range(self.num_player):
            self.player_system_state.append(str(i+self.offset_position) + " " +
                                            str(self.offset_career + self.card_to_player[i]))
            self.player_nlp_state.append("game_start "+self.current_names[i])

        # game state
        self.current_kill = -1
        self.current_heal = -1
        self.current_poison = -1
        self.current_guard = -1
        self.leader = -1
        self.hunter_kill = False
        self.hunter_idx = -1

        self.ready = True

    def get_names(self):
        return(copy.copy(self.current_names))

    def whole_village_info(self,text):
        language_list = text.split(" ")
        #use relative position in the language part
        for player_id in self.alive_list:
            current_language_list = []
            for token in language_list:
                if token in self.player_pos:
                    current_language_list.append(str(self.current_names[int(token)-self.offset_position]))
                else:
                    current_language_list.append(token)
            self.player_nlp_state[player_id] +=" "+" ".join(current_language_list)