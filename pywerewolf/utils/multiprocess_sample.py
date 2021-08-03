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

from pywerewolf.strategy.dict_headtoken_generator import dict_headtoken_generator
from pywerewolf.strategy.bert_headtoken_generator import bert_headtoken_generator
from pywerewolf.strategy.bert_a_generator import bert_a_generator
from pywerewolf.strategy.bert_q_generator import bert_q_generator
from pywerewolf.strategy.strategy_headtoken import strategy_headtoken

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

    #create manager first
    manager = eval(args.game_manager)(args.num_player,game_compose=args.game_compose)
    offset_game = manager.offset_game_all
    #create tokenizer
    tokenizer = tokenizer_base(manager.offset_special_token)

    #create head token generator
    if args.headtoken_generator == "dict_headtoken_generator":
        state_dict = np.load("headtoken_dict/"+args.state_list, allow_pickle=True).tolist()
        headtoken_generator = dict_headtoken_generator(state_dict, manager.offset_game_all)
    elif args.headtoken_generator == "bert_headtoken_generator":
        headtoken_generator = bert_headtoken_generator(headtoken_model,tokenizer,device)
    else:
        raise RuntimeError("unknown headtoken generator type")

    #create a generator
    a_generator = eval(args.a_generator)(a_model,tokenizer,device)

    #create q generator
    q_generator = eval(args.q_generator)(q_model, tokenizer, device)

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
            iter_string = "tqdm(range(args.game_per_sample//args.num_sampler))"
        else:
            iter_string = "range(args.game_per_sample//args.num_sampler)"

        game_record = []
        for idx in eval(iter_string):
            manager.reset()

            status = "continue"
            actions = [0., 0., 0., 0.]
            one_game = []
            while status == "continue":
                sys_s, nlp_s, act_mask, act_type, reward, status, ct = manager.step(actions)
                # actions = rand_strategy(act_mask,act_type)
                # compute actions
                actions = []
                response = []
                for i in range(manager.num_player):
                    act, rep = strategy.action(sys_s[i], nlp_s[i], act_mask, act_type[i])
                    actions.append(act)
                    response.append(rep)
                one_game.append([sys_s, nlp_s, act_mask, act_type, reward, status, ct, actions, response])
            game_record.append(one_game)
        count_decay += 1
        queue_out.put(game_record)

        #decay epsilon
        strategy.set_epsilon(sig_decay(count_decay,
                                     (1-args.game_epsilon_scale)*args.game_epsilon,
                                     args.game_halfpoint,
                                     args.game_epsilon_scale*args.game_epsilon,
                                     args.game_temper))

def start_samplegame_group(headtoken_model,a_model,q_model,queue_in_list,queue_out_list,device,args):
    process_pool = []
    for proc_idx in range(args.num_sampler):
        process_pool.append(mp.Process(target=samplegame_agent, args=(headtoken_model,a_model,q_model,
                                                                     proc_idx,queue_in_list[proc_idx],
                                                                     queue_out_list[proc_idx],device,args,)))
    for proc_idx in range(args.num_sampler):
        process_pool[proc_idx].start()
    return(process_pool)

def end_samplegame_group(process_pool):
    for proc in process_pool:
        proc.join()
    for proc in process_pool:
        proc.close()