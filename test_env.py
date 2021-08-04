import pywerewolf
import numpy as np
import copy
from tqdm import tqdm

num_player = 4
num_vocb = 16
num_names = 10
device = "cpu"

play_one_game = False
gen_headtoken = True
num_game = 10

name_headtoken = "1w_1p_2v_headtoken"
save_headtoken_dict = "headtoken_dict/"

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def main():
    game_compose = [0,0,1,7]

    # manager = pywerewolf.werewolf_env.werewolf_manager_base(num_player,game_compose = game_compose)

    manager = pywerewolf.werewolf_env.werewolf_manager_cyclic(num_player,game_compose=game_compose)

    # manager = pywerewolf.werewolf_env.werewolf_manager_named(num_player,num_names,
    #                                                          game_compose=game_compose)

    #total num of token in game
    offset_game = manager.offset_game_all
    #tokenizer used in q and a headtoken
    tokenizer = pywerewolf.utils.tokenizer_base(manager.offset_special_token)
    #a and q generator
    headtoken_generator = pywerewolf.strategy.empty_headtoken_generator()
    a_generator = pywerewolf.strategy.random_a_generator(num_player,num_vocb)
    #q generator
    q_generator = pywerewolf.strategy.random_q_generator(num_player,num_vocb)
    strategy = pywerewolf.strategy.strategy_headtoken(device,a_generator,q_generator,offset_game,headtoken_generator,eta=0.0)

    if play_one_game:
        manager.reset()
        status ="continue"
        actions = None

        while status =="continue":
            sys_s, nlp_s, act_mask, act_type, reward, status,ct = manager.step(actions)
            # actions = rand_strategy(act_mask,act_type)
            #compute actions
            actions = []
            for i in range(num_player):
                act,rep = strategy.action(sys_s[i], nlp_s[i], act_mask, act_type[i])
                actions.append(act)

            print("----------------------------------------------")
            print(sys_s)
            print(nlp_s)
            print(act_mask)
            print(act_type)
            print(reward)
            print(status)
            print(ct)

    state_dict = {}
    status_static = np.array([0,0,0],dtype=float)

    if gen_headtoken:
        for game_idx in tqdm(range(num_game)):
            manager.reset()
            status = "continue"
            actions = None
            # one_game = []
            while status == "continue":
                sys_s, nlp_s, act_mask, act_type, reward, status, ct = manager.step(actions)
                # actions = rand_strategy(act_mask,act_type)
                # compute actions
                actions = []
                response = []
                for i in range(num_player):
                    act,rep = strategy.action(sys_s[i], nlp_s[i], act_mask, act_type[i])
                    actions.append(act)
                    response.append(rep)
                    if act_type[i]!="none":
                        if sys_s[i] not in state_dict.keys():
                            state_dict.update({copy.copy(sys_s[i]):copy.copy(act_type[i])})
                # one_game.append([sys_s,nlp_s,act_mask,act_type,reward,status,ct,actions,response])


            if "were" in status:
                status_static[0] +=1
            elif "tie" in status:
                status_static[1] +=1
            elif "vil" in status:
                status_static[2] += 1
            else:
                raise RuntimeError("unknown status")
        for game in tqdm(game_record):
            store_data(act_state_dict,statement_state_dict,act_state_dict_sl,statement_state_dict_sl,game)

        print(status_static/num_game)
        print("find keys in total %d"%(len(state_dict.keys())))

        print(act_state_dict)

        # np.save(save_headtoken_dict+name_headtoken,state_dict)

    # print(manager.get_names())

if __name__ == "__main__":
    main()