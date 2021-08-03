import pywerewolf
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch
from tqdm import tqdm
import socket
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random
# from multiprocessing import Process, Queue
import time
import pandas as pd
import copy
start=0
end=50
num_player=4

#explore 1 . lr  2. memory size

parser = argparse.ArgumentParser(description='werewolfrl')

#RL env settings
parser.add_argument('--episode_per_update', type=int, default=64)
parser.add_argument('--num_process_game', type=int, default=4)
parser.add_argument('--batch_size_SL', type=int, default=16)
parser.add_argument('--batch_size_RL', type=int, default=16)
parser.add_argument('--iterstart_memorysize_SL', type=int, default=128)
parser.add_argument('--iterstart_memorysize_RL', type=int, default=128)
parser.add_argument('--SL_size', type=int, default=4096)
parser.add_argument('--RL_size', type=int, default=4096)
parser.add_argument('--split_keys', type=int, default=5)

#if no memory bank is used then all data from current sampling will be used
parser.add_argument('--use_targetnet', type=bool, default=False)
parser.add_argument('--update_target', type=int, default=1000)
#env settings
parser.add_argument('--game_eta', type=int, default=0.1)
parser.add_argument('--game_epsilon', type=float, default=0.4)
parser.add_argument('--game_epsilon_scale', type=float, default=0.1)
parser.add_argument('--game_halfpoint',type=int, default=100000)
parser.add_argument('--game_temper',type=float,default=500.)
#save settings
parser.add_argument('--save_update', type=int, default=1000)
parser.add_argument('--max_update', type=int, default=2000000)
#game settings
parser.add_argument('--world_size', type=int, default=4)
parser.add_argument('--game_compose', type=list, default=[0,0,1,7])
parser.add_argument('--num_player', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=16)
#training settings
parser.add_argument('--use_ema_model', type=bool, default=False)
parser.add_argument('--state_list',  type=str,  default="state_keys_1w_2v_1p.npy")
parser.add_argument('--memory_bank',  type=str,  default="memory_1w_2v_1p_vocb_16.npy")
# parser.add_argument('--state_list',  type=str,  default="state_keys_1w_3v_simple.npy")
# parser.add_argument('--memory_bank',  type=str,  default="memory_1w_3v_vocb_1.npy")

parser.add_argument('--actor_ema', type=float, default=1e-3)
parser.add_argument('--critic_q_ema', type=float, default=1e-2)
parser.add_argument('--loss_ema', type=float, default=1e-2)
parser.add_argument('--lr_act', type=float, default=1e-5)
parser.add_argument('--lr_q', type=float, default=1e-5)
parser.add_argument('--slow_rate', type=float, default=1.)
parser.add_argument('--warmup', type=bool, default=False)
parser.add_argument('--warmup_iter', type=int, default=2000)
parser.add_argument('--lr_epsilon_scale', type=float, default=0.5)
parser.add_argument('--lr_halfpoint',type=int, default=100000)
parser.add_argument('--lr_temper',type=float,default=2000.)
parser.add_argument('--seed', type=int, default=1234)
#model parameters
parser.add_argument('--hidden_size', type=int, default=96)
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_attention_heads', type=int, default=6)
parser.add_argument('--bert_intermediate_size', type=int, default=192)
parser.add_argument('--mlp_intermediate_size', type=int, default=4096)
parser.add_argument('--cri_intermediate_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.)
#training parameter
parser.add_argument('--aligned_size', type=int, default=48)
parser.add_argument('--gamma', type=float, default=1.0)
# parser.add_argument('--save_dir', type=str, default="model_small_a3c")
parser.add_argument('--save_dir', type=str, default="model_96_4_6_192_4096_2vil_1pro_1were_nfsp_stage_nlp_lr1e_5_1e_5_s64_b16_16_m_4096_4096_proc_4_sl1_rl1_opt_Adam_Adam_vocab_16_slow_rate_1_state_headtoken_loss_mean_split_eps_exp2")
parser.add_argument('--load_dir', type=str, default="pretain_3vil_1were_96_4_6_192_exp1")
parser.add_argument('--max_grad_norm_q', type=float, default=7)
parser.add_argument('--max_grad_norm_a', type=float, default=7)
parser.add_argument('--normalize_loss', type=float, default=30)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--load_episode', type=int, default=0)
# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
# os.environ['CUDA_VISIBLE_DEVICES']="4,5,6,7"
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def main(args):
    pid = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]

    hostfile = "tcp/dist_url_" + jobid + ".txt"

    ip = socket.gethostbyname(socket.gethostname())
    port = find_free_port()
    dist_url = None

    if pid == 0:
        dist_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(dist_url)
    else:
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, "r") as f:
            dist_url = f.read()

    mp.spawn(model_parallel, args=(pid, dist_url,args),
             nprocs=args.world_size,
             join=True)

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def setup(rank, dist_url,world_size,name='nccl'):
    dist.init_process_group(name, init_method=dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def sig_decay(iter,delta,half_point,low_bound,Temperature):
    return(delta/(1+np.exp((iter-half_point)/Temperature))+low_bound)

def samplegame_agent(actor_agent,critic_q_agent,args,pro_id,device,queue_in,queue_out,iter_num):
    print("agent %d of device %d is now running"%(pro_id,device),flush=True)
    #set random factor for current process
    np.random.seed(device+args.seed*pro_id+iter_num)
    torch.manual_seed(device+args.seed*pro_id+iter_num)
    random.seed(device+args.seed*pro_id+iter_num)

    #load the state key
    state_keys = np.load(args.state_list, allow_pickle=True).tolist()
    #create the local actor
    # strategy_list = \
    #     [pywerewolf.deepmodel.split_stage_nlp_headtoken_strategy.split_stage_nlp_headtoken_strategy(actor_agent,critic_q_agent,
    #                                                                             args.num_player,state_keys,
    #                                                                             device,
    #                                                                             eta=args.game_eta,
    #                                                                             epsilon=args.game_epsilon)
    #                                                                             for i in range(args.num_player)]

    strategy_list = \
        [pywerewolf.deepmodel.split_stage_nlp_headtoken_strategy_v2.split_stage_nlp_headtoken_strategy(actor_agent,critic_q_agent,
                                                                                args.num_player,state_keys,
                                                                                device,
                                                                                eta=args.game_eta,
                                                                                epsilon=args.game_epsilon)
                                                                                for i in range(args.num_player)]

    # multiple self player manager for para sampling of game
    manager_selfpaly = pywerewolf.werewolf_env.manager_split_stage_nlp_headtoken.werewolf_manager(args.num_player, strategy_list,
                                                                                        game_compose = args.game_compose)
    count_decay = 0
    while (True):
        memory = []
        cmd_state = queue_in.get()
        if cmd_state[0] == "done":
            break

        actor_agent.eval()
        critic_q_agent.eval()

        if pro_id == 0 and device==0:
            # l2_q = 0.
            # # l2 regu
            # for p in critic_q_agent.parameters():
            #     l2_q += torch.mean(p.pow(2))
            # print("in sampling l2 is %f"%(l2_q.detach().cpu().numpy()),flush=True)
            print("sampling game under latest policy", flush=True)
            iter_string = "tqdm(range(args.episode_per_update//args.num_process_game))"
        else:
            iter_string = "range(args.episode_per_update//args.num_process_game)"

        manager_selfpaly.set_training()

        for idx in eval(iter_string):
            manager_selfpaly.reset()
            round, status = manager_selfpaly.run_game(pid=pro_id)
            memory.append(record_game(manager_selfpaly, status))
        count_decay += 1
        queue_out.put(memory)
        for idx in range(args.num_player):
            manager_selfpaly.player_list[idx].strategy.set_epsilon(sig_decay(count_decay,
                                                                             (1-args.game_epsilon_scale)*args.game_epsilon,
                                                                             args.game_halfpoint,
                                                                             args.game_epsilon_scale*args.game_epsilon,
                                                                             args.game_temper))


def start_samplegame_group(actor,critic_q,args,device,queue_in_list,queue_out_list,iter_num):
    process_pool = []
    for proc_idx in range(args.num_process_game):
        process_pool.append(mp.Process(target=samplegame_agent, args=(actor,critic_q,
                                                                        args,proc_idx,device,
                                                                        queue_in_list[proc_idx],
                                                                        queue_out_list[proc_idx],
                                                                        iter_num,)))
    for proc_idx in range(args.num_process_game):
        process_pool[proc_idx].start()
    return(process_pool)

def end_samplegame_group(process_pool):
    for proc in process_pool:
        proc.join()
    for proc in process_pool:
        proc.close()


def model_parallel(rank,pid,dist_url,args):
    #gpu： gpu number of current machine
    #rank: rank number of whole training world
    gpu = rank
    rank = pid*8+gpu
    #set up the current training env
    device = gpu if torch.cuda.is_available() else torch.device('cpu')
    print("current rank is %d"%(rank),flush=True)
    setup(rank,dist_url, args.world_size)

    #create the local actor
    state_keys = np.load(args.state_list, allow_pickle=True).tolist()
    ordered_key_list = [key for key in state_keys.keys()]
    ordered_key_list.sort()



    # if gpu==0:
    #     print(state_keys,flush=True)

    #tokenizer
    tokenizer_obj = pywerewolf.deepmodel.tokenizer_stage_nlp.Tokenizer_headtoken(num_player=args.num_player,
                                                                                vocb_size = args.vocab_size,
                                                                                num_state=len(ordered_key_list))

    #set the random set for current numpy random seed
    np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)

    if gpu==0:
        pywerewolf.utils.dump.create_files(args.save_dir)

    #config for the model
    config = pywerewolf.deepmodel.model.BertConfig(vocab_size=tokenizer_obj.total_vocab_size,
                                                   hidden_size=args.hidden_size,
                                                   num_hidden_layers=args.num_hidden_layers,
                                                   num_attention_heads=args.num_attention_heads,
                                                   intermediate_size=args.bert_intermediate_size,
                                                   hidden_dropout_prob=args.dropout,
                                                   attention_probs_dropout_prob=args.dropout)

    #global actor used to train model accross device
    actor = pywerewolf.deepmodel.model_stage_nlp_headtoken.stage_strategy(config,args.num_player,
                                                                device = device,
                                                                num_words=args.vocab_size,
                                                                mlp_intermedia=args.mlp_intermediate_size).to(device)

    critic_q = pywerewolf.deepmodel.model_stage_nlp_headtoken.stage_strategy(config,args.num_player,
                                                                   device=device,
                                                                    num_words=args.vocab_size,
                                                                    mlp_intermedia=args.mlp_intermediate_size).to(device)

    actor.share_memory()
    critic_q.share_memory()

    if args.load_model:
        state_dict = torch.load("ckpt/" + args.load_dir + "/act_episode"+str(args.load_episode),
                                map_location=torch.device(device))
        actor.load_state_dict(state_dict)
        state_dict = torch.load("ckpt/" + args.load_dir + "/cri_q_episode"+str(args.load_episode),
                                map_location=torch.device(device))
        critic_q.load_state_dict(state_dict)

    actor_ddp = DDP(actor, device_ids=[device])
    critic_q_ddp = DDP(critic_q,device_ids=[device])

    #target model, used only for q function
    actor_ema = pywerewolf.deepmodel.model_stage_nlp_headtoken.stage_strategy(config,args.num_player,
                                                                device=device,
                                                                num_words=args.vocab_size,
                                                                mlp_intermedia=args.mlp_intermediate_size).to(device)

    critic_q_target = pywerewolf.deepmodel.model_stage_nlp_headtoken.stage_strategy(config,args.num_player,
                                                                          device=device,
                                                                        num_words=args.vocab_size,
                                                                        mlp_intermedia=args.mlp_intermediate_size).to(device)

    #syn the initial paramters of global model
    for parameter in actor_ddp.parameters():
        torch.distributed.broadcast(parameter,0,async_op=False)

    for parameter in critic_q_ddp.parameters():
        torch.distributed.broadcast(parameter,0,async_op=False)

    #set the current target model as global
    critic_q_target.load_state_dict(critic_q_ddp.module.state_dict())
    actor_ema.load_state_dict(actor_ddp.module.state_dict())

    # #create self play manager

    #create the parameter group
    actor_slow_update = []
    actor_fast_update = []
    for name,p in actor_ddp.named_parameters():
        if "module_list" in name and "fc2.bias" in name:
            actor_fast_update.append(p)
        else:
            actor_slow_update.append(p)

    critic_slow_update = []
    critic_fast_update = []
    for name,p in critic_q_ddp.named_parameters():
        if "module_list" in name and "fc2.bias" in name:
            critic_fast_update.append(p)
        else:
            critic_slow_update.append(p)


    optimizer_act = optim.Adam([{"params": actor_slow_update,"lr": args.lr_act*args.slow_rate},
                                  {"params": actor_fast_update, "lr": args.lr_act}],
                             betas=(0.9, 0.999), eps=1e-6, weight_decay=0., amsgrad=False)

    optimizer_cri_q = optim.Adam([{"params": critic_slow_update,"lr": args.lr_q*args.slow_rate},
                                  {"params": critic_fast_update, "lr": args.lr_q}],
                             betas=(0.9, 0.999), eps=1e-6, weight_decay=0., amsgrad=False)


    # main RL loop
    update_idx = 0

    #monitor variable
    loss_a_mean = 0.
    loss_q_mean = 0.

    #load from a presampled memory buffer to avoid the initial sampling stage
    memory_act,memory_nlp = np.load(args.memory_bank, allow_pickle=True).tolist()
    #memory used for RL and SL
    state_act_SL = {}
    state_nlp_SL = {}
    for name in ordered_key_list:
        if state_keys[name]=="act":
            state_act_SL.update({name:[copy.copy(memory_act[name][0])+
                                       [None for i in range(args.SL_size-memory_act[name][5])],
                                       copy.copy(memory_act[name][1]) +
                                       [None for i in range(args.SL_size-memory_act[name][5])],
                                       copy.copy(memory_act[name][2]) +
                                       [None for i in range(args.SL_size - memory_act[name][5])],
                                       copy.copy(memory_act[name][3]) +
                                       [None for i in range(args.SL_size - memory_act[name][5])],
                                       memory_act[name][5],memory_act[name][5]]})


        else:
            state_nlp_SL.update({name:[copy.copy(memory_nlp[name][0])+
                                      [None for i in range(args.SL_size-memory_nlp[name][5])],
                                      copy.copy(memory_nlp[name][1]) +
                                      [None for i in range(args.SL_size - memory_nlp[name][5])],
                                      copy.copy(memory_nlp[name][2]) +
                                      [None for i in range(args.SL_size - memory_nlp[name][5])],
                                      copy.copy(memory_nlp[name][3]) +
                                      [None for i in range(args.SL_size - memory_nlp[name][5])],
                                      memory_nlp[name][5],memory_nlp[name][5]]})

    state_act_RL = {}
    state_nlp_RL = {}

    for name in ordered_key_list:
        if state_keys[name]=="act":
            #data, last pos flag, count
            #data: nlp info, nlp mask, action id, action mask, reward
            state_act_RL.update({name:[copy.copy(memory_act[name][0])+
                                       [None for i in range(args.RL_size-memory_act[name][5])],
                                       copy.copy(memory_act[name][1]) +
                                       [None for i in range(args.RL_size - memory_act[name][5])],
                                       copy.copy(memory_act[name][2]) +
                                       [None for i in range(args.RL_size - memory_act[name][5])],
                                       copy.copy(memory_act[name][3]) +
                                       [None for i in range(args.RL_size - memory_act[name][5])],
                                       copy.copy(memory_act[name][4]) +
                                       [None for i in range(args.RL_size - memory_act[name][5])],
                                       memory_act[name][5],memory_act[name][5]]})
        else:
            state_nlp_RL.update({name:[copy.copy(memory_nlp[name][0])+
                                       [None for i in range(args.RL_size-memory_nlp[name][5])],
                                       copy.copy(memory_nlp[name][1]) +
                                       [None for i in range(args.RL_size - memory_nlp[name][5])],
                                       copy.copy(memory_nlp[name][2]) +
                                       [None for i in range(args.RL_size - memory_nlp[name][5])],
                                       copy.copy(memory_nlp[name][3]) +
                                       [None for i in range(args.RL_size - memory_nlp[name][5])],
                                       copy.copy(memory_nlp[name][4]) +
                                       [None for i in range(args.RL_size - memory_nlp[name][5])],
                                       memory_nlp[name][5],memory_nlp[name][5]]})

    #generate the split of keys
    act_split = [[] for i in range(args.split_keys)]
    nlp_split = [[] for i in range(args.split_keys)]

    count_split = 0
    for key in state_act_RL.keys():
        act_split[count_split].append(key)
        count_split =(count_split+1)%args.split_keys

    count_split = 0
    for key in state_nlp_RL.keys():
        nlp_split[count_split].append(key)
        count_split =(count_split+1)%args.split_keys

    queue_in_list = [mp.Queue(1) for i in range(args.num_process_game)]
    queue_out_list = [mp.Queue(1) for i in range(args.num_process_game)]
    proc_pool = start_samplegame_group(actor,critic_q,args, device, queue_in_list, queue_out_list, 0)

    while update_idx <args.max_update:
        #compute the current dict first
        if args.use_ema_model:
            moving_mean(actor_ema, actor_ddp.module, args.actor_ema)
            moving_mean(critic_q_target, critic_q_ddp.module, args.critic_q_ema)
        else:
            # actor_ema.load_state_dict(actor_ddp.module.state_dict())
            # critic_q_target.load_state_dict(critic_q_ddp.module.state_dict())
            pass

        #sample game using multiprocess
        memory = []
        # state_dict_q = critic_q_target.state_dict()
        for q_id in range(args.num_process_game):
            queue_in_list[q_id].put(("sample",None,None))
        for q_id in range(args.num_process_game):
            memory += queue_out_list[q_id].get()

        # for q_id in range(args.num_process_game):
        #     queue_in_list[q_id].put(("done",None,None))
        # end_samplegame_group(proc_pool)
        #the progress the updating idx
        update_idx +=1

        #process the result of self play and store them
        if gpu==0:
            print("",flush=True)
            print("process self play game data", flush=True)
            iter_str = "tqdm(range(len(memory)))"
        else:
            iter_str = "range(len(memory))"

        for iter in eval(iter_str):
            pywerewolf.utils.sample_stage_nlp.store_game_nfsp_stage_nlp(memory,
                                                                         state_act_SL,
                                                                         state_nlp_SL,
                                                                         state_act_RL,
                                                                         state_nlp_RL,
                                                                         iter,
                                                                         args.num_player,
                                                                         tokenizer_obj,
                                                                         args.aligned_size,
                                                                         args.gamma)

            SL_all = 0
            RL_all = 0
            sample_exists_SL = 1
            sample_exists_RL = 1

            #get SL sampling information
            for name in ordered_key_list:
                if name in state_act_SL.keys():
                    SL_all += state_act_SL[name][4]
                    # tmp_sample_exists = tmp_sample_exists if state_act_SL[name][4]<1 else 1
                    sample_exists_SL = sample_exists_SL if state_act_SL[name][4] > 0 else 0
                else:
                    SL_all += state_nlp_SL[name][4]
                    # tmp_sample_exists = tmp_sample_exists if state_nlp_SL[name][4]<1 else 1
                    sample_exists_SL = sample_exists_SL if state_nlp_SL[name][4] > 0 else 0

            #get RL sampling information
            for name in ordered_key_list:
                if name in state_act_RL.keys():
                    RL_all += state_act_RL[name][5]
                    # tmp_sample_exists = tmp_sample_exists if state_act_RL[name][5]<1 else 1
                    sample_exists_RL = sample_exists_RL if state_act_RL[name][5] >0 else 0
                else:
                    RL_all += state_nlp_RL[name][5]
                    # tmp_sample_exists = tmp_sample_exists if state_nlp_RL[name][5]<1 else 1
                    sample_exists_RL = sample_exists_RL if state_nlp_RL[name][5] >0 else 0

        # current min length of SL RL
        SL_exists = torch.tensor(sample_exists_SL, dtype=torch.int64).to(device)
        RL_exists = torch.tensor(sample_exists_RL, dtype=torch.int64).to(device)
        torch.distributed.all_reduce(SL_exists, op=torch.distributed.ReduceOp.MIN, async_op=False)
        torch.distributed.all_reduce(RL_exists, op=torch.distributed.ReduceOp.MIN, async_op=False)

        SL_size = torch.tensor(SL_all, dtype=torch.int64).to(device)
        RL_size = torch.tensor(RL_all, dtype=torch.int64).to(device)
        torch.distributed.all_reduce(SL_size, op=torch.distributed.ReduceOp.MIN, async_op=False)
        torch.distributed.all_reduce(RL_size, op=torch.distributed.ReduceOp.MIN, async_op=False)

        if gpu==0:
            print("",flush=True)

            print("current data set size %d, %d from %d"%(SL_size.detach().cpu().numpy(),
                                                          RL_size.detach().cpu().numpy(),
                                                          len(memory)),flush=True)

            print("current exists %d, %d" % (SL_exists.detach().cpu().numpy(),
                                             RL_exists.detach().cpu().numpy()), flush=True)

        critic_q_ddp.train()
        actor_ddp.train()

        #sample from recorded memory and update policy and q function
        if gpu==0:
            iter_str_SL = "tqdm(range(args.split_keys))"
            iter_str_RL = "tqdm(range(args.split_keys))"
        else:
            iter_str_SL = "range(args.split_keys)"
            iter_str_RL = "range(args.split_keys)"

        if gpu==0:
            print("",flush=True)
            print("update policy network",flush=True)


        for iter in eval(iter_str_SL):
            #start update for policy network
            if SL_exists==1 and SL_size>=args.iterstart_memorysize_SL:

                input_type_SL = []
                input_state_SL = []
                input_nlp_info_SL = []
                input_mask_SL = []
                input_action_ids_SL = []
                input_action_mask_SL = []

                #split key for act and nlp for sampling
                for key in act_split[iter]:
                    sample_num = min(args.batch_size_SL,state_act_SL[key][4])
                    sample_index = random.sample(range(state_act_SL[key][4]),sample_num)

                    type = state_keys[key]
                    headtoken = str(ordered_key_list.index(key)+args.num_player+args.vocab_size)
                    headtoken_ids = tokenizer_obj([headtoken])[0]

                    input_type_SL += [type for i in sample_index]
                    input_state_SL += [headtoken_ids for i in sample_index]
                    input_nlp_info_SL += [state_act_SL[key][0][i] for i in sample_index]
                    input_mask_SL += [state_act_SL[key][1][i] for i in sample_index]
                    input_action_ids_SL += [[state_act_SL[key][2][i]] for i in sample_index]
                    input_action_mask_SL += [state_act_SL[key][3][i] for i in sample_index]

                for key in nlp_split[iter]:
                    sample_num = min(args.batch_size_SL,state_nlp_SL[key][4])
                    sample_index = random.sample(range(state_nlp_SL[key][4]),sample_num)

                    type = state_keys[key]
                    headtoken = str(ordered_key_list.index(key)+args.num_player+args.vocab_size)
                    headtoken_ids = tokenizer_obj([headtoken])[0]

                    input_type_SL += [type for i in sample_index]
                    input_state_SL += [headtoken_ids for i in sample_index]
                    input_nlp_info_SL += [state_nlp_SL[key][0][i] for i in sample_index]
                    input_mask_SL += [state_nlp_SL[key][1][i] for i in sample_index]
                    input_action_ids_SL += [[state_nlp_SL[key][2][i]] for i in sample_index]
                    input_action_mask_SL += [state_nlp_SL[key][3][i] for i in sample_index]

                assert len(input_state_SL)>0
                #training the policy network
                type_SL = input_type_SL
                headtoken_SL = torch.tensor(input_state_SL,dtype=torch.int64).to(device)
                nlp_info_SL = torch.tensor(input_nlp_info_SL, dtype=torch.int64).to(device)
                type_nlp_SL = torch.cat((headtoken_SL,nlp_info_SL),dim=1)

                b = len(input_type_SL)
                leading_one = torch.ones((b,1),dtype=torch.int64).to(device)
                mask_SL = torch.tensor(input_mask_SL, dtype=torch.int64).to(device)
                type_mask_SL = torch.cat((leading_one,mask_SL),dim=1)

                action_ids_SL = torch.tensor(input_action_ids_SL,dtype=torch.int64).to(device)

                # actors special input
                actor_output,actor_h,act_mask = actor_ddp(type_SL, type_nlp_SL, attention_mask=type_mask_SL)
                assert len(actor_output)==2
                # backward and update
                gather_index_act = action_ids_SL[act_mask]
                gather_index_nlp = action_ids_SL[~act_mask]
                num_act = act_mask.sum().to(torch.int).detach().cpu().numpy()
                act_mask_tensor = torch.tensor(input_action_mask_SL[0:num_act],dtype=torch.float32).to(device)

                #compute loss
                prob_act = F.softmax(actor_output[0]+act_mask_tensor,dim=-1)
                prob_nlp = F.softmax(actor_output[1],dim=-1)

                prob_act_selected = prob_act.gather(1,gather_index_act)
                prob_nlp_selected = prob_nlp.gather(1,gather_index_nlp)

                #log loss for act
                # loss_act = -torch.log(prob_act_selected).sum()/args.batch_size_SL
                # loss_nlp = -torch.log(prob_nlp_selected).sum()/args.batch_size_SL

                loss_act = -torch.log(prob_act_selected).mean()
                loss_nlp = -torch.log(prob_nlp_selected).mean()
                loss_a = loss_act+loss_nlp+(prob_act*prob_act).sum()*0.+(prob_nlp*prob_nlp).sum()*0.


                # update policy parameters
                optimizer_act.zero_grad()
                loss_a.backward()
                #compute the current norm of gradient
                # torch.nn.utils.clip_grad_norm_(actor_ddp.parameters(), args.max_grad_norm_a)
                optimizer_act.step()

                if gpu == 0:
                    loss_a_mean = loss_a.detach().cpu().numpy() * args.loss_ema + loss_a_mean * (1 - args.loss_ema)

        if gpu==0:
            print("",flush=True)
            print("update q function",flush=True)

        for iter in eval(iter_str_RL):
            #start update for q value function
            if RL_exists==1 and RL_size >= args.iterstart_memorysize_RL:

                input_type1_RL = []
                input_state1_RL = []
                input_nlp_info1_RL = []
                input_mask1_RL = []
                input_action_ids_s1_RL = []
                input_action_mask_s1_RL = []
                input_reward_RL = []
                for key in act_split[iter]:
                    sample_num = min(args.batch_size_RL, state_act_RL[key][5])
                    sample_index = random.sample(range(state_act_RL[key][5]), sample_num)

                    type = state_keys[key]
                    headtoken = str(ordered_key_list.index(key)+args.num_player+args.vocab_size)
                    headtoken_ids = tokenizer_obj([headtoken])[0]

                    input_type1_RL += [type for i in sample_index]
                    input_state1_RL += [headtoken_ids for i in sample_index]
                    input_nlp_info1_RL += [state_act_RL[key][0][i] for i in sample_index]
                    input_mask1_RL += [state_act_RL[key][1][i] for i in sample_index]
                    input_action_ids_s1_RL += [[state_act_RL[key][2][i]] for i in sample_index]
                    input_action_mask_s1_RL += [state_act_RL[key][3][i] for i in sample_index]
                    input_reward_RL += [state_act_RL[key][4][i] for i in sample_index]

                for key in nlp_split[iter]:
                    sample_num = min(args.batch_size_RL, state_nlp_RL[key][5])
                    sample_index = random.sample(range(state_nlp_RL[key][5]), sample_num)

                    type = state_keys[key]
                    headtoken = str(ordered_key_list.index(key)+args.num_player+args.vocab_size)
                    headtoken_ids = tokenizer_obj([headtoken])[0]

                    input_type1_RL += [type for i in sample_index]
                    input_state1_RL += [headtoken_ids for i in sample_index]
                    input_nlp_info1_RL += [state_nlp_RL[key][0][i] for i in sample_index]
                    input_mask1_RL += [state_nlp_RL[key][1][i] for i in sample_index]
                    input_action_ids_s1_RL += [[state_nlp_RL[key][2][i]] for i in sample_index]
                    input_action_mask_s1_RL += [state_nlp_RL[key][3][i] for i in sample_index]
                    input_reward_RL += [state_nlp_RL[key][4][i] for i in sample_index]

                assert len(input_state1_RL)>0
                #training the q network
                type1_RL = input_type1_RL
                headtoken1_RL = torch.tensor(input_state1_RL,dtype=torch.int64).to(device)
                nlp_info1_RL = torch.tensor(input_nlp_info1_RL, dtype=torch.int64).to(device)
                type_nlp1_RL = torch.cat((headtoken1_RL,nlp_info1_RL),dim=1)

                b = len(input_type1_RL)
                leading_one = torch.ones((b,1),dtype=torch.int64).to(device)
                mask1_RL = torch.tensor(input_mask1_RL, dtype=torch.int64).to(device)
                type_mask1_RL = torch.cat((leading_one,mask1_RL),dim=1)

                action_ids_s1_RL = torch.tensor(input_action_ids_s1_RL, dtype=torch.int64).to(device)
                action_mask_s1_RL = input_action_mask_s1_RL

                #average the weight by this coefficent
                critic_s1_q,critic_h,act_mask = critic_q_ddp(type1_RL, type_nlp1_RL, attention_mask=type_mask1_RL)
                estimate = torch.tensor(input_reward_RL, dtype=torch.float32).to(device).view(-1)

                assert len(critic_s1_q)==2
                #compute loss
                target_act = estimate[act_mask]
                target_nlp = estimate[~act_mask]

                index_act = action_ids_s1_RL[act_mask]
                index_nlp = action_ids_s1_RL[~act_mask]
                pred_q_act = critic_s1_q[0].gather(1,index_act)
                pred_q_nlp = critic_s1_q[1].gather(1, index_nlp)

                # loss_q_act = (0.5*(pred_q_act.view(-1)-target_act)**2).sum()/args.batch_size_RL
                # loss_q_nlp = (0.5*(pred_q_nlp.view(-1)-target_nlp)**2).sum()/args.batch_size_RL
                loss_q_act = (0.5*(pred_q_act.view(-1)-target_act)**2).mean()
                loss_q_nlp = (0.5*(pred_q_nlp.view(-1)-target_nlp)**2).mean()

                loss_q = loss_q_act+loss_q_nlp + \
                         (critic_s1_q[0]*critic_s1_q[0]).sum()*0. + (critic_s1_q[1]*critic_s1_q[1]).sum()*0.

                # update parameters
                optimizer_cri_q.zero_grad()
                loss_q.backward()

                # torch.nn.utils.clip_grad_norm_(critic_q_ddp.parameters(), args.max_grad_norm_q)
                optimizer_cri_q.step()

                if gpu == 0:
                    # print("in training l2 is %f"%(l2_q.detach().cpu().numpy()),flush=True)
                    loss_q_mean = loss_q.detach().cpu().numpy() * args.loss_ema + loss_q_mean * (1 - args.loss_ema)
        #update lr
        if args.warmup:
            tmp = min((1-args.slow_rate)*update_idx/args.warmup_iter + args.slow_rate,1.)
            scale_list = [tmp, 1]
        else:
            scale_list = [args.slow_rate,1]
        g_id = 0
        for g in optimizer_act.param_groups:
            g['lr']=sig_decay(update_idx,(1-args.lr_epsilon_scale)*args.lr_act*scale_list[g_id],
                                            args.lr_halfpoint,
                                            args.lr_epsilon_scale*args.lr_act*scale_list[g_id],
                                            args.lr_temper)
            g_id +=1

        g_id = 0
        for g in optimizer_cri_q.param_groups:
            g['lr']=sig_decay(update_idx,(1-args.lr_epsilon_scale)*args.lr_q*scale_list[g_id],
                                            args.lr_halfpoint,
                                            args.lr_epsilon_scale*args.lr_q*scale_list[g_id],
                                            args.lr_temper)
            g_id += 1

        if rank==0:
            print("",flush=True)
            lr_a = []
            lr_q = []
            for g in optimizer_act.param_groups:
                lr_a.append(g['lr'])
            for g in optimizer_cri_q.param_groups:
                lr_q.append(g['lr'])
            pywerewolf.utils.dump.dumpy_current(args.save_dir, [loss_a_mean,loss_q_mean])
            print("update %d lr_a is %f %f,lr_q is %f %f"%(update_idx,lr_a[0],lr_a[1],lr_q[0],lr_q[1]),flush=True)
            print("at update %d mean loss_a %f"%(update_idx,loss_a_mean),flush=True)
            print("at update %d mean loss_q %f" % (update_idx, loss_q_mean), flush=True)
            # print("at update %d mean cosin_loss_a %f" % (update_idx, cosin_loss_a_ema), flush=True)
            # print("at update %d mean cosin_loss_q %f" % (update_idx, cosin_loss_q_ema), flush=True)

        if rank==0 and update_idx%args.save_update==0:
            episode = update_idx*args.world_size*args.episode_per_update
            torch.save(actor_ddp.module.state_dict(), "ckpt/" + args.save_dir + "/act_episode" + str(episode))
            torch.save(critic_q_ddp.module.state_dict(),"ckpt/" + args.save_dir + "/cri_q_episode" + str(episode))
            torch.save(critic_q_target.state_dict(), "ckpt/" + args.save_dir + "/cri_q_target_episode" + str(episode))
            #dump the current memory
            if not os.path.exists("memory/"+args.save_dir):
                os.mkdir("memory/"+args.save_dir)
            np.save("memory/"+args.save_dir+"/sl_act_dict"+str(episode),state_act_SL)
            np.save("memory/" + args.save_dir + "/sl_nlp_dict" + str(episode), state_nlp_SL)
            np.save("memory/" + args.save_dir + "/rl_act_dict" + str(episode), state_act_RL)
            np.save("memory/" + args.save_dir + "/rl_nlp_dict" + str(episode), state_nlp_RL)


    for q_id in range(args.num_process_game):
        queue_in_list[q_id].put(("done", None, None))
    end_samplegame_group(proc_pool)


def moving_mean(model_ema,model,ema_coef):
    for p_ema,p in zip(model_ema.parameters(),
                       model.parameters()):
        p_ema.data.copy_((1-ema_coef)*p_ema + p*ema_coef)

def test_game(manager,num_game,use_tqdm,composite):
    #compute the mean win rate of agent
    werewolf_win = 0

    if use_tqdm:
        range_str = "tqdm(range(num_game))"
    else:
        range_str = "range(num_game)"
    for i in eval(range_str):
        manager.reset(game_compose=composite)
        round,status = manager.run_game()
        if 'were' in status:
            werewolf_win+=1
    return(werewolf_win)



def record_game(manager,status):
    game_state = []
    game_info = []
    game_action = []
    game_reward = []
    for idx in range(len(manager.player_list)):
        game_state.append(manager.player_list[idx].get_state_bank())
        game_info.append(manager.player_list[idx].get_info_bank())
        game_action.append(manager.player_list[idx].get_action_bank())
        game_reward.append(manager.player_list[idx].get_final_reward())
    return([game_state,game_info,game_action,game_reward,[status,[manager.player_list[j].career for j in range(num_player)]]])

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)