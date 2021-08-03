import hashlib
import copy

class hash_store_data_fast(object):
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

                #replace the direct data string with hashkey
                hash_key = hashlib.sha1(sys1.encode("utf-8")).hexdigest()
                if hash_key not in act_state_dict.keys():
                    eval(pos_str).update({hash_key:0})
                    eval(pos_sl_str).update({hash_key: 0})
                    eval(state_str).update({hash_key:[None for i in range(self.max_data_per_key)]})
                    eval(state_sl_str).update({hash_key:[None for i in range(self.max_data_per_key)]})

                pos = eval(pos_str)[hash_key]
                if act_type == "act":
                    self.act_state_total = (self.act_state_total + 1) if act_state_dict[hash_key][-1] == None \
                        else (self.act_state_total + 0)
                else:
                    self.statement_state_total = (self.statement_state_total + 1) if statement_state_dict[hash_key][
                                                                                         -1] == None \
                        else (self.statement_state_total + 0)

                eval(state_str)[hash_key][pos] = {"sys1": copy.copy(sys1),
                                              "sys2": copy.copy(sys2),
                                              "nlp1": copy.copy(nlp1),
                                              "nlp2": copy.copy(nlp2),
                                              "act_mask": copy.copy(act_mask),
                                              "act_type": copy.copy(act_type),
                                              "r_imme": copy.copy(r_imme),
                                              "r_finnal": copy.copy(finnal_reward[player_id]),
                                              "act_id": copy.copy(action_id)}

                eval(pos_str)[hash_key] = (eval(pos_str)[hash_key] + 1) % self.max_data_per_key

                if one_game[act_position[player_id][act_id]][8][player_id] == "br":
                    pos_sl = eval(pos_sl_str)[hash_key]
                    if act_type== "act":
                        self.act_state_total = (self.act_state_total + 1) if act_state_dict[hash_key][-1] == None \
                            else (self.act_state_total + 0)
                    else:
                        self.statement_state_total = (self.statement_state_total + 1) if statement_state_dict[hash_key][
                                                                                             -1] == None \
                            else (self.statement_state_total + 0)
                    eval(state_sl_str)[hash_key][pos_sl] = {"sys1": copy.copy(sys1),
                                                      "sys2": copy.copy(sys2),
                                                      "nlp1": copy.copy(nlp1),
                                                      "nlp2": copy.copy(nlp2),
                                                      "act_mask": copy.copy(act_mask),
                                                      "act_type": copy.copy(act_type),
                                                      "r_imme": copy.copy(r_imme),
                                                      "r_finnal": copy.copy(finnal_reward[player_id]),
                                                      "act_id": copy.copy(action_id)}

                    eval(pos_sl_str)[hash_key] = (eval(pos_sl_str)[hash_key] + 1) % self.max_data_per_key