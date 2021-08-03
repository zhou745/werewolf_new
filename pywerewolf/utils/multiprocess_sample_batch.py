import numpy as np
import torch
import random
import copy
import torch.multiprocessing as mp
from tqdm import tqdm

from pywerewolf.werewolf_env.werewolf_manager_cyclic import werewolf_manager_cyclic
from pywerewolf.werewolf_env.werewolf_manager_named import werewolf_manager_named
from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base

from pywerewolf.utils.tokenizer import tokenizer_base


from pywerewolf.strategy_batch.bert_headtoken_generator_batch import bert_headtoken_generator_batch
from pywerewolf.strategy_batch.bert_a_generator_batch import bert_a_generator_batch
from pywerewolf.strategy_batch.bert_q_generator_batch import bert_q_generator_batch
from pywerewolf.strategy_batch.strategy_headtoken_batch import strategy_headtoken_batch

def sig_decay(iter,delta,half_point,low_bound,Temperature):
    return(delta/(1+np.exp((iter-half_point)/Temperature))+low_bound)

def samplegame_agent(headtoken_model,a_model,q_model,
                     pro_id,queue_in,queue_out,
                     device,args):

    device_num = device if isinstance(device,int) else -1
    print("agent %d of device %d is now running"%(pro_id,device_num),flush=True)
    #set random factor for current process
    np.random.seed(device_num+args.seed*pro_id+1)
    torch.manual_seed(device_num+args.seed*pro_id+1)
    random.seed(device_num+args.seed*pro_id+1)

    game_compose_process = copy.copy(args.game_compose)
    #create manager first
    manager_list = []
    for mid in range(args.num_manager):
        manager_list.append(eval(args.game_manager)(args.num_player,game_compose=args.game_compose,
                                                                    game_step = args.game_step))

    offset_game = manager_list[0].offset_game_all
    #create tokenizer
    tokenizer = tokenizer_base(manager_list[0].offset_special_token)

    #create head token generator
    if args.headtoken_generator == "bert_headtoken_generator_batch":
        headtoken_generator = bert_headtoken_generator_batch(headtoken_model,tokenizer,device,aligned_size=args.aligned_size)
    else:
        raise RuntimeError("unknown headtoken generator type")

    #create a generator
    a_generator = eval(args.a_generator)(a_model,tokenizer,device,aligned_size=args.aligned_size)

    #create q generator
    q_generator = eval(args.q_generator)(q_model, tokenizer, device,aligned_size=args.aligned_size)

    #create strategy
    strategy = eval(args.strategy)(device,a_generator,q_generator,offset_game,headtoken_generator,eta=args.game_eta,
                                                                                                  epsilon = args.game_epsilon)

    count_decay = 0
    while (True):
        cmd_state = queue_in.get()
        if cmd_state[0] == "done":
            break

        if headtoken_model!=None:
            headtoken_model.eval()
        if a_model!=None:
            a_model.eval()
        if q_model!=None:
            q_model.eval()

        if pro_id == 0 and device in [0,"cpu"]:
            print("sampling game under latest policy", flush=True)
            iter_string = "tqdm(range(args.game_per_sample//args.num_sampler//args.num_manager))"
        else:
            iter_string = "range(args.game_per_sample//args.num_sampler//args.num_manager)"

        game_record = []
        action_list = [None for ma in range(len(manager_list))]
        response_list = [None for ma in range(len(manager_list))]
        for idx in eval(iter_string):
            running_game = [ma for ma in range(len(manager_list))]

            #reset all manager
            for manager_idx in range(len(manager_list)):
                random.shuffle(game_compose_process)
                manager_list[manager_idx].reset(career_assign=game_compose_process)

            status = "continue"
            game_list = [[] for ma in range(len(manager_list))]

            while len(running_game) >0:
                current_ended_game = []

                sys_s_b = []
                nlp_s_b = []
                act_mask_b = []
                act_type_b = []

                batch_index_list = []
                for ma in running_game:
                    sys_s, nlp_s, act_mask, act_type, reward, status, ct = manager_list[ma].step(action_list[ma])
                    #record current ended game
                    if status != "continue":
                        current_ended_game.append(ma)
                    action_list[ma] = [0 for i in range(manager_list[ma].num_player)]
                    response_list[ma] = ["none" for i in range(manager_list[ma].num_player)]

                    for i in range(manager_list[ma].num_player):
                        if act_type[i] !="none":
                            batch_index_list.append((ma,i))
                            sys_s_b.append(sys_s[i])
                            nlp_s_b.append(nlp_s[i])
                            if act_mask!=None:
                                act_mask_b.append(act_mask)
                            else:
                                act_mask_b.append([1 for j in range(manager_list[ma].num_player)])
                            act_type_b.append(act_type[i])

                    #record most info later mofify act
                    game_list[ma].append([sys_s, nlp_s, act_mask, act_type, reward, status, ct, action_list[ma], response_list[ma]])

                #compute actions
                if len(sys_s_b)>0:
                    sys_s_b = np.array(sys_s_b)
                    nlp_s_b = np.array(nlp_s_b)
                    act_mask_b =np.array(act_mask_b)
                    act_type_b = np.array(act_type_b)

                    #compute action with batch
                    act_list, rep_list = strategy.action(sys_s_b, nlp_s_b, act_mask_b, act_type_b)
                    #modify precomputed act and rep
                    for index in range(len(batch_index_list)):
                        ma = batch_index_list[index][0]
                        i = batch_index_list[index][1]
                        game_list[ma][-1][7][i] = act_list[index]
                        game_list[ma][-1][8][i] = rep_list[index]
                        action_list[ma][i] = act_list[index]
                        response_list[ma][i] = rep_list[index]
                for ended_idx in current_ended_game:
                    running_game.remove(ended_idx)


            game_record += game_list
        count_decay += 1
        queue_out.put(game_record)

        #decay epsilon
        strategy.set_epsilon(sig_decay(count_decay,
                                     (1-args.game_epsilon_scale)*args.game_epsilon,
                                     args.game_halfpoint,
                                     args.game_epsilon_scale*args.game_epsilon,
                                     args.game_temper))

def start_samplegame_group_batch(headtoken_model,a_model,q_model,queue_in_list,queue_out_list,device,args):
    process_pool = []
    for proc_idx in range(args.num_sampler):
        process_pool.append(mp.Process(target=samplegame_agent, args=(headtoken_model,a_model,q_model,
                                                                     proc_idx,queue_in_list[proc_idx],
                                                                     queue_out_list[proc_idx],device,args,)))
    for proc_idx in range(args.num_sampler):
        process_pool[proc_idx].start()
    return(process_pool)

def end_samplegame_group_batch(process_pool):
    for proc in process_pool:
        proc.join()
    for proc in process_pool:
        proc.close()