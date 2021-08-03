import numpy as np
import copy
import random


"""
villager 0 Prophet 1 witch 2
hunter 3 idiot 4 guard 5
kignt 6 

werewolf 7 wolfking 8
whitewolf 9 woflbeauty 10 snowwolf 11
"""

#embeding ordering
# player_id  career_type  vocb_tokens  system_tokens

#default game composition
game_make_dict={4:[0,0,1,7],5:[0,1,2,7,7],6:[0,0,1,5,7,7],7:[0,0,0,1,5,7,7],8:[0,0,1,2,5,7,7,7]}

default_game_step = ["check_werewolf_team","werewolf_kill","prophet_check","summerize_night",
                     "plyer_last_statement","make_statement_inturn","vote_for_one","plyer_last_statement"]

class werewolf_manager_base(object):
    def __init__(self,num_player,game_compose=None,
                                 game_step=default_game_step,
                                 player_size_reserve = 100,
                                 player_career_reserve = 100,
                                 player_vocb_reserve = 100,
                                 special_token_reserve=100):

        #flag ready to run env
        self.ready = False

        #game para used in the env
        self.num_player = num_player
        self.game_compose = game_compose
        self.game_step = game_step

        #get the current career type
        if game_compose is None:
            self.card_to_player = game_make_dict[num_player].copy()
        else:
            self.card_to_player = game_compose.copy()
        #shuffle the career card
        random.shuffle(self.card_to_player)

        #progress state
        self.current_progress = 0

        #set the game
        self.player_system_state = []
        self.player_nlp_state = []
        self.alive_list = []
        self.current_dead = []
        #a player in type_id must be alive
        self.type_id = [[] for i in range(12)]

        #game state
        self.current_kill = -1
        self.current_heal = -1
        self.current_poison = -1
        self.current_guard = -1
        self.leader = -1

        self.hunter_kill = False
        self.hunter_idx = -1

        self.current_time = "day 0"

        #record the token reserve size
        self.player_size_reserve = player_size_reserve
        self.player_career_reserve = player_career_reserve
        self.player_vocb_reserve = player_vocb_reserve
        self.special_token_reserve = special_token_reserve

        #record the start position for all token
        self.offset_position = 0
        self.offset_career = self.player_size_reserve
        self.offset_vocb = self.offset_career + player_career_reserve
        self.offset_special_token = self.offset_vocb + player_vocb_reserve
        self.offset_game_all = self.offset_special_token + special_token_reserve

    def get_reserve_dict(self):
        reserve_dict = {"player_size_reserve":self.player_size_reserve,
                        "player_career_reserve":self.player_career_reserve,
                        "player_vocb_reserve":self.player_vocb_reserve,
                        "special_token_reserve":self.special_token_reserve
                        }
        return(reserve_dict)

    def step(self,action_list):
        assert self.ready==True
        #get the function to progress the game
        action_fun = eval("self."+self.game_step[self.current_progress])
        #progress the game
        self.progress_idx()

        #get the game evolution
        system_state, nlp_state, action_mask,action_type,reward,status,current_time = action_fun(action_list)

        return(copy.copy(system_state),
               copy.copy(nlp_state),
               copy.copy(action_mask),
               copy.copy(action_type),
               copy.copy(reward),
               copy.copy(status),
               copy.copy(current_time))

    def progress_idx(self):
        self.current_progress = (self.current_progress+1)%len(self.game_step)
        if self.current_progress == 0:
            self.current_progress +=1

    def progress_time(self):
        time_list = self.current_time.split(" ")
        if "day"==time_list[0]:
            time_list[0] = "night"
        else:
            time_list[0] = "day"
            time_list[1] = str(int(time_list[1])+1)
        self.current_time = " ".join(time_list)

    def reset(self,career_assign=None):
        #reset the game

        #reset the game step back to 0
        self.current_progress = 0
        self.current_time = "day 0"

        #reset game progress
        self.player_system_state = []
        self.player_nlp_state = []
        self.alive_list = []
        self.current_dead = []
        self.type_id = [[] for i in range(12)]

        #reset the game career
        if career_assign==None:
            random.shuffle(self.card_to_player)
        else:
            self.card_to_player = copy.copy(career_assign)

        #set the career type
        for i in range(self.num_player):
            #append player i to type self.card_to_player[i]
            self.type_id[self.card_to_player[i]].append(i)
            self.alive_list.append(i)

        #initialize all the game state
        for i in range(self.num_player):
            self.player_system_state.append(str(i+self.offset_position)+" "
                                            +str(self.card_to_player[i]+self.offset_career))
            self.player_nlp_state.append("game_start")

        #game state
        self.current_kill = -1
        self.current_heal = -1
        self.current_poison = -1
        self.current_guard = -1
        self.leader = -1
        self.hunter_kill = False
        self.hunter_idx = -1

        self.ready = True

    #only run onece
    def check_werewolf_team(self,action):
        werewolf_list = []

        #loop through all the player information offset by position offset
        for player_id in self.type_id[7]:
            werewolf_list.append(str(player_id+self.offset_position))

        werewolf_list.sort()
        check_werewolf = " ".join(werewolf_list)
        check_werewolf = "check_werewolf_team "+check_werewolf

        #progress the player state acordingly
        for player_id in self.type_id[7]:
            self.player_system_state[player_id] +=" "+check_werewolf



        #create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        #create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        #not terminate reward 0
        reward = [0. for player_id in range(self.num_player)]

        status = "continue"
        #compute action mask for current state
        return(self.player_system_state,
               self.player_nlp_state,
               action_mask,
               action_type,
               reward,
               status,
               self.current_time)

    #runing repeatedly
    def werewolf_kill(self,action):

        #progress time
        self.progress_time()

        #decide one to kill
        vote_list = []
        for player_id in self.type_id[7]:
            vote_list.append(action[player_id])

        # randomly select one target from the vote_list
        self.current_kill = random.sample(vote_list, 1)[0]  # position id

        # progress the state offset py position
        for player_id in self.type_id[7]:
            self.player_system_state[player_id] +=" werewolf_agreed_kill "+str(self.current_kill+self.offset_position)

        #create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        #create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        #not terminate reward 0
        reward = [0. for player_id in range(self.num_player)]

        status = "continue"

        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def prophet_check(self,action):
        # let the prophet to check
        for player_id in self.type_id[1]:
            check_idx = action[player_id]

            #offset by position
            prophet_checked = "prophet_check_result "
            if check_idx in self.type_id[7]:
                prophet_checked += str(check_idx+self.offset_position) + " bad"
            else:
                prophet_checked += str(check_idx+self.offset_position) + " good"

            self.player_system_state[player_id] += " "+prophet_checked

        # create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        # create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        # not terminate reward 0
        reward = [0. for player_id in range(self.num_player)]

        status = "continue"

        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def guard_select(self,action):

        for player_id in self.type_id[5]:
            #offset by position
            guard_select = "guard_select "+str(action[player_id])
            
            self.current_guard = action[player_id]

            self.player_system_state[player_id] += " "+guard_select

        # create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        # create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        # not terminate reward 0
        reward = [0. for player_id in range(self.num_player)]

        status = "continue"

        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def summerize_night(self,action):
        #progress time
        self.progress_time()

        #reset the deadlist to empty
        self.current_dead = []

        # kill success only when witch not heal and guard not guard
        if self.current_heal != self.current_kill and self.current_guard != self.current_kill:
            self.current_dead.append(self.current_kill)

        # witch poison is always effective
        if self.current_poison not in self.current_dead and self.current_poison in self.alive_list:
            self.current_dead.append(self.current_poison)

        # announce the dead
        if len(self.current_dead)>0:
            dead_list = [str(player_id+self.offset_position) for player_id in self.current_dead]
            dead_state = "dead_people "+" ".join(dead_list)
        else:
            dead_state = "dead_people none"

        #progress the state to all player
        self.whole_village_progress(dead_state)

        #remove all the dead players
        for player_id in self.current_dead:
            self.alive_list.remove(player_id)
            # remove it from its career list
            self.type_id[self.card_to_player[player_id]].remove(player_id)

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
            reward = [1 if self.card_to_player[player_id]==7 else -1
                      for player_id  in range(self.num_player)]
        else:
            reward = [-1 if self.card_to_player[player_id]==7 else 1
                      for player_id  in range(self.num_player)]

        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def plyer_last_statement(self,action):
        if len(self.current_dead)==0:
            plyer_last_statement = "plyer_last_statement none"
            self.whole_village_info(plyer_last_statement)
        else:
            for player_id in self.current_dead:
                plyer_last_statement = "plyer_last_statement "+str(player_id+self.offset_position)+" "
                #
                plyer_last_statement+=str(action[player_id]+self.offset_vocb)
                # self.whole_village_info(plyer_last_statement)
                self.whole_village_info(plyer_last_statement)

        # create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        # create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        reward = [0. for player_id in range(self.num_player)]

        status = "continue"

        #everytime last statement said empty the dead people list
        self.current_dead = []

        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def make_statement_inturn(self,action):
        #!ToDo currently the statement start from a random alive player, later villager leader will choose where to start

        #statement in turn
        statement_list = []
        for player_id in self.alive_list:
            # player_idx = self.alive_list[(start+direction*idx)%player_num]
            statement_list.append("player_statement "+str(player_id+self.offset_position)+
                                  " "+str(action[player_id]+self.offset_vocb))

        statement_info = " ".join(statement_list)
        # print(statement_info,flush=True)
        self.whole_village_progress("statement collected")
        self.whole_village_info(statement_info)

        # create all the action mask
        action_mask = self.gen_actmask(self.game_step[self.current_progress])

        # create all the action type
        action_type = self.gen_acttype(self.game_step[self.current_progress])

        reward = [0. for player_id in range(self.num_player)]

        status = "continue"
        # compute action mask for current state
        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def vote_for_one(self,action):
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
        voted_out = "voted_out " + str(voted+self.offset_position)
        self.current_dead.append(voted)
        self.whole_village_progress(voted_out)

        #remove the dead player
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
            reward = [1 if self.card_to_player[player_id]==7 else -1
                      for player_id  in range(self.num_player)]
        else:
            reward = [-1 if self.card_to_player[player_id]==7 else 1
                      for player_id  in range(self.num_player)]

        return (self.player_system_state,
                self.player_nlp_state,
                action_mask,
                action_type,
                reward,
                status,
                self.current_time)

    def gen_actmask(self,step_name):
        if step_name in ["werewolf_kill","prophet_check","vote_for_one"]:
            action_mask = [0. if player_id in self.alive_list else float("-inf")
                           for player_id in range(self.num_player)]
        elif step_name == "guard_select":
            action_mask = [0. if player_id in self.alive_list else float("-inf")
                           for player_id in range(self.num_player)]
            if self.current_guard>-0.5:
                action_mask[self.current_guard] = float("-inf")
        else:
            action_mask = None

        return(action_mask)

    def gen_acttype(self,step_name):
        if step_name=="werewolf_kill":
            action_type = ["act" if player_id in self.type_id[7] else "none"
                           for player_id in range(self.num_player)]
        elif step_name=="prophet_check":
            action_type = ["act" if player_id in self.type_id[1] else "none"
                           for player_id in range(self.num_player)]
        elif step_name=="summerize_night":
            action_type = ["none"
                           for player_id in range(self.num_player)]
        elif step_name=="plyer_last_statement":
            action_type = ["statement" if player_id in self.current_dead else "none"
                           for player_id in range(self.num_player)]
        elif step_name=="make_statement_inturn":
            action_type = ["statement" if player_id in self.alive_list else "none"
                           for player_id in range(self.num_player)]
        elif step_name=="vote_for_one":
            action_type = ["act" if player_id in self.alive_list else "none"
                           for player_id in range(self.num_player)]
        elif step_name == "guard_select":
            action_type = ["act" if player_id in self.type_id[5] else "none"
                           for player_id in range(self.num_player)]            
        return(action_type)

    def whole_village_progress(self,text):
        for player_id in self.alive_list:
            self.player_system_state[player_id] += " "+text

    def whole_village_info(self,text):
        for player_id in self.alive_list:
            self.player_nlp_state[player_id] +=" "+text

    def check_terminal(self):

        remain_werewolf = 0
        remain_human = 0
        for idx in range(5):
            remain_werewolf += len(self.type_id[idx + 7])
        for idx in range(7):
            remain_human += len(self.type_id[idx])

        if remain_human > 0 and remain_werewolf == 0:
            return ("villager_win")
        elif remain_werewolf > 0 and remain_human == 0:
            return ("werewolf_win")
        elif remain_werewolf == 0 and remain_human == 0:
            return ("tied_game")
        else:
            return ("continue")