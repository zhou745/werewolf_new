import copy
import numpy as np
import pywerewolf

game_step = ["check_werewolf_team","werewolf_kill","prophet_check","guard_select","summerize_night",
                     "plyer_last_statement","make_statement_inturn","vote_for_one","plyer_last_statement"]

num_player = 6

#night time return a list of possible tokens,

def werewolf_kill(players):
    key_str_list = []
    for player_id in players:
        key_str_list.append(" werewolf_agreed_kill "+str(player_id))
    return(key_str_list)

def prophet_check(players):
    key_str_list = []
    for player_id in players:
        key_str_list.append(" prophet_check_result "+str(player_id)+" bad")
        key_str_list.append(" prophet_check_result " + str(player_id) + " good")
    return(key_str_list)

def guard_select(players):
    key_str_list = []
    for player_id in players:
        key_str_list.append(" guard_select "+str(player_id))
    return(key_str_list)

#day time return, all player share
def day_time(players):
    #announce daed
    key_list = []

    #dead keys
    dead_key_list = []
    for player_id in players:
        dead_key_list.append(" dead_people "+str(player_id))
    dead_key_list.append(" dead_people none")

    key_list += copy.copy(dead_key_list)

    #statement keys
    statemet_key_list = []
    for key in dead_key_list:
        statemet_key_list.append(key+" statement collected")

    key_list += copy.copy(statemet_key_list)
    #voted keys
    voted_key_list = []
    for key in statemet_key_list:
        for player_id in players:
            voted_key_list.append(key+" voted_out "+str(player_id))

    key_list += copy.copy(voted_key_list)
    return (key_list)

#gen initial keys
# 2 werewolf
def init_keys(players,career_types = [100,101,105,107]):
    key_list = []
    for player_id in players:
        for career in career_types:
            key_list.append(str(player_id) + " " + str(career))
    #special treatment for werewolf
    for player_a_id in players:
        for player_b_id in players:
            if player_a_id==player_b_id:
                continue
            key_list.append(str(player_a_id) + " 107 check_werewolf_team " + str(player_a_id) + " " +str(player_b_id))
            key_list.append(str(player_b_id) + " 107 check_werewolf_team " + str(player_a_id) + " " + str(player_b_id))
    return(key_list)

def main(num_player,game_step):
    players = [i for i in range(num_player)]
    key_list = []
    if "werewolf_kill" in game_step:
        key_list += werewolf_kill(players)

    if "prophet_check" in game_step:
        key_list += prophet_check(players)

    if "guard_select" in game_step:
        key_list += guard_select(players)

    key_list += day_time(players)
    key_list +=init_keys(players)

    key_list.sort()
    key_dict = {}
    for i in range(len(key_list)):
        key_dict.update({key_list[i]:i})
    key_dict.update({"":len(key_list)})
    print("%d %d"%(len(key_list),len(key_dict.keys())))

    np.save("headtoken_dict/2w_1g_1p_2v_timed",key_dict)

if __name__ == "__main__":
    main(num_player,game_step)