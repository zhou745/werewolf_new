import pywerewolf
import numpy as np
import copy
from tqdm import tqdm

num_player = 6
num_vocb = 16
num_names = 10
device = "cpu"

play_one_game = False
check_generated_keys = True
num_game = 10000

name_headtoken = "1w_1p_2v_headtoken"
save_headtoken_dict = "headtoken_dict/"

game_step = ["check_werewolf_team","werewolf_kill","prophet_check","guard_select","summerize_night",
                     "plyer_last_statement","make_statement_inturn","vote_for_one","plyer_last_statement"]

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def main():
    game_compose = [0,0,1,5,7,7]

    # manager = pywerewolf.werewolf_env.werewolf_manager_base(num_player,game_compose = game_compose,
    #                                                                    game_step = game_step)

    # manager = pywerewolf.werewolf_env.werewolf_manager_cyclic(num_player,game_compose=game_compose)

    # manager = pywerewolf.werewolf_env.werewolf_manager_named(num_player,num_names,
    #                                                          game_compose=game_compose)

    # manager = pywerewolf.werewolf_env.werewolf_manager_timed(num_player, game_compose=game_compose,
    #                                                                      game_step=game_step)

    manager = pywerewolf.werewolf_env.werewolf_manager_timed_cyclic(num_player, game_compose=game_compose,
                                                                                game_step=game_step,
                                                                                max_time=10)

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
    status_static = np.array([0,0,0],dtype=float)

    if check_generated_keys:
        load_tokens = np.load("headtoken_dict/2w_1g_1p_2v_timed.npy",allow_pickle=True).tolist()

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
                    for key in sys_s[i]:
                        try:
                            assert key in load_tokens.keys()
                        except:
                            raise RuntimeError("\'"+key+"\'" + " not found in token_list")

            if "were" in status:
                status_static[0] +=1
            elif "tie" in status:
                status_static[1] +=1
            elif "vil" in status:
                status_static[2] += 1
            else:
                raise RuntimeError("unknown status")

        print(status_static/num_game)

    # print(manager.get_names())

if __name__ == "__main__":
    main()