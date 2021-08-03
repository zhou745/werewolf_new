import copy

class store_data_base(object):
    def __init__(self,max_data_per_key):
        self.max_data_per_key = max_data_per_key

        self.act_state_pos_dict = {}
        self.statement_state_pos_dict = {}

        self.act_state_sl_pos_dict = {}
        self.statement_sl_state_pos_dict = {}

        self.act_state_total = 0
        self.statement_state_total = 0

        self.act_state_sl_total = 0
        self.statement_sl_state_total = 0

    def __call__(self,act_state_dict,statement_state_dict,
                 act_state_dict_sl,statement_state_dict_sl,one_game):
        #action is stored in the order of 0:system state, 1:nlp state,2:act_mask,
        #                                 3:act_type, 4:immediate reward,5:status,
        #                                 6:game time,7:action token,8:response_type

        finnal_reward = one_game[-1][4]
        num_player = len(one_game[-1][0])

        #system state
        sys1_list = [[] for i in range(num_player)]
        #nlp state
        nlp1_list = [[] for i in range(num_player)]
        #act mask
        act_mask_list = [[] for i in range(num_player)]
        #act type
        act_type_list = [[] for i in range(num_player)]
        #immediate reward
        r_imme_list = [[] for i in range(num_player)]
        #game time
        time_list = [[] for i in range(num_player)]
        #action_list
        action_list = [[] for i in range(num_player)]
        #response list
        response_list = [[] for i in range(num_player)]
        #finnal reward list
        r_final_list = [[] for i in range(num_player)]

        for step_idx in range(len(one_game)):
            for player_id in range(num_player):
                #the game must be in continue state for a act to be valid
                if one_game[step_idx][3][player_id]!="none" and one_game[step_idx][5]=="continue":
                    sys1_list[player_id].append(one_game[step_idx][0][player_id])
                    nlp1_list[player_id].append(one_game[step_idx][1][player_id])
                    act_mask_list[player_id].append(one_game[step_idx][2])
                    act_type_list[player_id].append(one_game[step_idx][3][player_id])
                    r_imme_list[player_id].append(one_game[step_idx+1][4][player_id])
                    time_list[player_id].append(one_game[step_idx][6])
                    action_list[player_id].append(one_game[step_idx][7][player_id])
                    response_list[player_id].append(one_game[step_idx][8][player_id])
                    r_final_list[player_id].append(finnal_reward[player_id])

        #store all the ordered game in the memory dict
        for player_id in range(num_player):
            for state_idx in range(len(sys1_list[player_id])):
                sys1 = sys1_list[player_id][state_idx]
                sys2 = sys1_list[player_id][state_idx + 1] if state_idx + 1 < len(sys1_list[player_id]) else None

                nlp1 = nlp1_list[player_id][state_idx]
                nlp2 = nlp1_list[player_id][state_idx + 1] if state_idx + 1 < len(nlp1_list[player_id]) else None

                #decide which dict to use
                if act_type_list[player_id][state_idx]=="act":
                    pos_str = "self.act_state_pos_dict"
                    pos_sl_str = "self.act_state_sl_pos_dict"

                    state_str = "act_state_dict"
                    state_sl_str = "act_state_dict_sl"
                else:
                    pos_str = "self.statement_state_pos_dict"
                    pos_sl_str = "self.statement_sl_state_pos_dict"

                    state_str = "statement_state_dict"
                    state_sl_str = "statement_state_dict_sl"

                if sys1 not in act_state_dict.keys():
                    eval(pos_str).update({sys1:0})
                    eval(pos_sl_str).update({sys1: 0})
                    eval(state_str).update({sys1:[None for i in range(self.max_data_per_key)]})
                    eval(state_sl_str).update({sys1:[None for i in range(self.max_data_per_key)]})

                pos = eval(pos_str)[sys1]
                if act_type_list[player_id][state_idx]=="act":
                    self.act_state_total = (self.act_state_total  + 1) if act_state_dict[sys1][-1] == None \
                                            else (self.act_state_total  + 0)
                else:
                    self.statement_state_total = (self.statement_state_total+1) if statement_state_dict[sys1][-1] == None \
                                            else (self.statement_state_total  + 0)

                eval(state_str)[sys1][pos] = {"sys1":copy.copy(sys1),
                                            "sys2":copy.copy(sys2),
                                            "nlp1":copy.copy(nlp1),
                                            "nlp2":copy.copy(nlp2),
                                            "act_mask":copy.copy(act_mask_list[player_id][state_idx]),
                                            "act_type":copy.copy(act_type_list[player_id][state_idx]),
                                            "r_imme":copy.copy(r_imme_list[player_id][state_idx]),
                                            "r_finnal":copy.copy(r_final_list[player_id][state_idx]),
                                            "act_id":copy.copy(action_list[player_id][state_idx])}

                eval(pos_str)[sys1] = (eval(pos_str)[sys1]+1)%self.max_data_per_key

                if response_list[player_id][state_idx] == "br":
                    pos_sl = eval(pos_sl_str)[sys1]
                    if act_type_list[player_id][state_idx] == "act":
                        self.act_state_total = (self.act_state_total + 1) if act_state_dict[sys1][-1] == None \
                            else (self.act_state_total + 0)
                    else:
                        self.statement_state_total = (self.statement_state_total + 1) if statement_state_dict[sys1][
                                                                                             -1] == None \
                            else (self.statement_state_total + 0)
                    eval(state_sl_str)[sys1][pos_sl] = {"sys1":copy.copy(sys1),
                                                        "sys2":copy.copy(sys2),
                                                        "nlp1":copy.copy(nlp1),
                                                        "nlp2":copy.copy(nlp2),
                                                        "act_mask":copy.copy(act_mask_list[player_id][state_idx]),
                                                        "act_type":copy.copy(act_type_list[player_id][state_idx]),
                                                        "r_imme":copy.copy(r_imme_list[player_id][state_idx]),
                                                        "r_finnal":copy.copy(r_final_list[player_id][state_idx]),
                                                        "act_id":copy.copy(action_list[player_id][state_idx])}

                    eval(pos_sl_str)[sys1] = (eval(pos_sl_str)[sys1] + 1) % self.max_data_per_key




class store_data_base_fast(object):
    def __init__(self,max_data_per_key):
        self.max_data_per_key = max_data_per_key

        self.act_state_pos_dict = {}
        self.statement_state_pos_dict = {}

        self.act_state_sl_pos_dict = {}
        self.statement_sl_state_pos_dict = {}

        self.act_state_total = 0
        self.statement_state_total = 0

        self.act_state_sl_total = 0
        self.statement_sl_state_total = 0

    def __call__(self,act_state_dict,statement_state_dict,
                 act_state_dict_sl,statement_state_dict_sl,one_game):
        #action is stored in the order of 0:system state, 1:nlp state,2:act_mask,
        #                                 3:act_type, 4:immediate reward,5:status,
        #                                 6:game time,7:action token,8:response_type

        finnal_reward = one_game[-1][4]
        num_player = len(one_game[-1][0])

        act_position = [[] for i in range(num_player)]
        #compute the act position flag first
        for player_id in range(num_player):
            for step_idx in range(len(one_game)):
                if one_game[step_idx][3][player_id]!="none" and one_game[step_idx][5]=="continue":
                    act_position[player_id].append(step_idx)

        #store the
        for player_id in range(num_player):
            for act_id in range(len(act_position[player_id])):
                sys1 = one_game[act_position[player_id][act_id]][0][player_id]
                sys2 = one_game[act_position[player_id][act_id+1]][0][player_id] if  act_id+1< len(act_position[player_id]) else None

                nlp1 = one_game[act_position[player_id][act_id]][1][player_id]
                nlp2 = one_game[act_position[player_id][act_id+1]][1][player_id] if  act_id+1< len(act_position[player_id]) else None

                act_mask = one_game[act_position[player_id][act_id]][2]
                act_type = one_game[act_position[player_id][act_id]][3][player_id]
                action_id = one_game[act_position[player_id][act_id]][7][player_id]

                r_imme = one_game[act_position[player_id][act_id]][4][player_id]

                if act_type=="act":
                    pos_str = "self.act_state_pos_dict"
                    pos_sl_str = "self.act_state_sl_pos_dict"

                    state_str = "act_state_dict"
                    state_sl_str = "act_state_dict_sl"
                else:
                    pos_str = "self.statement_state_pos_dict"
                    pos_sl_str = "self.statement_sl_state_pos_dict"

                    state_str = "statement_state_dict"
                    state_sl_str = "statement_state_dict_sl"

                if sys1 not in act_state_dict.keys():
                    eval(pos_str).update({sys1:0})
                    eval(pos_sl_str).update({sys1: 0})
                    eval(state_str).update({sys1:[None for i in range(self.max_data_per_key)]})
                    eval(state_sl_str).update({sys1:[None for i in range(self.max_data_per_key)]})

                pos = eval(pos_str)[sys1]
                if act_type == "act":
                    self.act_state_total = (self.act_state_total + 1) if act_state_dict[sys1][-1] == None \
                        else (self.act_state_total + 0)
                else:
                    self.statement_state_total = (self.statement_state_total + 1) if statement_state_dict[sys1][
                                                                                         -1] == None \
                        else (self.statement_state_total + 0)

                eval(state_str)[sys1][pos] = {"sys1": copy.copy(sys1),
                                              "sys2": copy.copy(sys2),
                                              "nlp1": copy.copy(nlp1),
                                              "nlp2": copy.copy(nlp2),
                                              "act_mask": copy.copy(act_mask),
                                              "act_type": copy.copy(act_type),
                                              "r_imme": copy.copy(r_imme),
                                              "r_finnal": copy.copy(finnal_reward[player_id]),
                                              "act_id": copy.copy(action_id)}

                eval(pos_str)[sys1] = (eval(pos_str)[sys1] + 1) % self.max_data_per_key

                if one_game[act_position[player_id][act_id]][8][player_id] == "br":
                    pos_sl = eval(pos_sl_str)[sys1]
                    if act_type== "act":
                        self.act_state_total = (self.act_state_total + 1) if act_state_dict[sys1][-1] == None \
                            else (self.act_state_total + 0)
                    else:
                        self.statement_state_total = (self.statement_state_total + 1) if statement_state_dict[sys1][
                                                                                             -1] == None \
                            else (self.statement_state_total + 0)
                    eval(state_sl_str)[sys1][pos_sl] = {"sys1": copy.copy(sys1),
                                                      "sys2": copy.copy(sys2),
                                                      "nlp1": copy.copy(nlp1),
                                                      "nlp2": copy.copy(nlp2),
                                                      "act_mask": copy.copy(act_mask),
                                                      "act_type": copy.copy(act_type),
                                                      "r_imme": copy.copy(r_imme),
                                                      "r_finnal": copy.copy(finnal_reward[player_id]),
                                                      "act_id": copy.copy(action_id)}

                    eval(pos_sl_str)[sys1] = (eval(pos_sl_str)[sys1] + 1) % self.max_data_per_key