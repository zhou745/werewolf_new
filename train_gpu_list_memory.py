import numpy as np
import argparse
import pywerewolf
import torch.multiprocessing as mp
from tqdm import tqdm
import torch
import os
import socket
import time
import torch.distributed as dist

from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base
from pywerewolf.werewolf_env.werewolf_manager_cyclic import werewolf_manager_cyclic
from pywerewolf.werewolf_env.werewolf_manager_named import werewolf_manager_named


parser = argparse.ArgumentParser(description='config_name')
parser.add_argument('--config_name', type=str)

os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"

def main(args):
    pid = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]
    config_training = np.load("training_config/"+args.config_name,allow_pickle=True).tolist()

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

    mp.spawn(model_parallel, args=(pid, dist_url,config_training),
             nprocs=config_training.world_size,
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

def model_parallel(rank,pid,dist_url,config_training):

    gpu = rank
    rank = pid*8+gpu
    #set up the current training env
    device = gpu if torch.cuda.is_available() else torch.device('cpu')
    print("current rank is %d"%(rank),flush=True)
    setup(rank,dist_url, config_training.world_size)

    #config for the model
    manager_fake = eval(config_training.game_manager)(config_training.num_player,game_compose=config_training.game_compose)
    tokenizer = pywerewolf.utils.tokenizer_base(manager_fake.offset_special_token)

    config_model = pywerewolf.deepmodel.BertConfig(vocab_size=manager_fake.offset_game_all,
                                                           hidden_size=config_training.hidden_size,
                                                           num_hidden_layers=config_training.num_hidden_layers,
                                                           num_attention_heads=config_training.num_attention_heads,
                                                           intermediate_size=config_training.bert_intermediate_size,
                                                           hidden_dropout_prob=config_training.dropout,
                                                           attention_probs_dropout_prob=config_training.dropout)

    # deepmodel_headtoken = None
    # deepmodel_a = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,config_training.vocab_size)
    # deepmodel_q = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,config_training.vocab_size)

    deepmodel_headtoken = pywerewolf.deepmodel.headtoken_model(config_model,config_training.mlp_intermediate_size).to(device)
    deepmodel_a = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,
                                                            config_training.vocab_size,config_training.mlp_intermediate_size).to(device)
    deepmodel_q = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,
                                                            config_training.vocab_size,config_training.mlp_intermediate_size).to(device)

    deepmodel_headtoken.share_memory()
    deepmodel_a.share_memory()
    deepmodel_q.share_memory()

    queue_in_list = [mp.Queue(1) for i in range(config_training.num_sampler)]
    queue_out_list = [mp.Queue(1) for i in range(config_training.num_sampler)]

    proc_pool = pywerewolf.utils.start_samplegame_group(deepmodel_headtoken,deepmodel_a,deepmodel_q,
                                                        queue_in_list, queue_out_list,device,config_training)


    # act_state_dict = {}
    # statement_state_dict = {}
    # act_state_dict_sl = {}
    # statement_state_dict_sl = {}

    act_state_dict = [None for i in range(config_training.memory_size)]
    statement_state_dict = [None for i in range(config_training.memory_size)]
    act_state_dict_sl = [None for i in range(config_training.memory_size)]
    statement_state_dict_sl = [None for i in range(config_training.memory_size)]

    #data store
    # store_data = pywerewolf.utils.store_data_base_fast(1024)
    store_data = pywerewolf.utils.list_store_data_fast(config_training.memory_size)

    sampler_rl = pywerewolf.utils.uniform_sampler(config_training.batch_size_RL)
    sampler_sl = pywerewolf.utils.uniform_sampler(config_training.batch_size_SL)

    augment = pywerewolf.utils.cyclic_player(manager_fake.offset_position,
                                                manager_fake.offset_career,
                                                manager_fake.offset_vocb,
                                                manager_fake.offset_special_token,
                                                manager_fake.offset_relative_pos,
                                                manager_fake.num_player)

    batch_collect = pywerewolf.utils.batch_collector(tokenizer,config_training.aligned_size,config_training.num_player)

    #main training loop
    for iter in range(config_training.max_update):
        memory = []
        #sampling once
        for q_id in range(config_training.num_sampler):
            queue_in_list[q_id].put(("sample", None, None))
        for q_id in range(config_training.num_sampler):
            memory += queue_out_list[q_id].get()

        #process game data
        if device==0:
            str_loop = "tqdm(memory)"
        else:
            str_loop = "memory"
        for game in eval(str_loop):
            store_data(act_state_dict, statement_state_dict, act_state_dict_sl, statement_state_dict_sl, game)

        deepmodel_headtoken.train()
        deepmodel_q.train()
        deepmodel_a.train()

        #sample data for training
        if store_data.statement_state_total>config_training.iterstart_memorysize_RL and \
           store_data.act_state_total>config_training.iterstart_memorysize_RL:
            #reset train

            sampled_act = sampler_rl(act_state_dict,store_data.act_state_total)
            sampled_statement = sampler_rl(statement_state_dict,store_data.statement_state_total)

            sampled_data = sampled_act + sampled_statement
            #augment data act
            augmented_sample_data= []
            for one_action in sampled_data:
                one_action_aug = augment(one_action)
                augmented_sample_data.append(one_action_aug)

            #collect batch
            batched_data = batch_collect(augmented_sample_data)

            #conver to cuda
            s1_cuda = torch.tensor(batched_data["s1"],dtype=torch.int64).to(device)
            s1_atten_mask = torch.tensor(batched_data["s1_atten_mask"],dtype=torch.int64).to(device)

            nlp1_cuda = torch.tensor(batched_data["nlp1"],dtype=torch.int64).to(device)
            nlp1_atten_mask = torch.tensor(batched_data["nlp1_atten_mask"],dtype=torch.int64).to(device)

            act_type = torch.tensor(batched_data["act_type"],dtype=torch.bool).to(device)
            #compute headtoken
            # headtoken = deepmodel_headtoken(s1_cuda,attention_mask=s1_atten_mask)
            # q_value = deepmodel_q(headtoken,nlp1_cuda,act_type,attention_mask=nlp1_atten_mask)

        #train action
        if store_data.statement_sl_state_total>config_training.iterstart_memorysize_SL and \
           store_data.act_state_sl_total>config_training.iterstart_memorysize_SL:

            sampled_act = sampler_sl(act_state_dict_sl,store_data.act_state_sl_total)
            sampled_statement = sampler_sl(statement_state_dict_sl,store_data.statement_sl_state_total)

            sampled_data = sampled_act + sampled_statement
            #augment data act
            augmented_sample_data= []
            for one_action in sampled_data:
                one_action_aug = augment(one_action)
                augmented_sample_data.append(one_action_aug)

            #collect batch
            batched_data = batch_collect(augmented_sample_data)

            #conver to cuda
            s1_cuda = torch.tensor(batched_data["s1"],dtype=torch.int64).to(device)
            s1_atten_mask = torch.tensor(batched_data["s1_atten_mask"],dtype=torch.int64).to(device)

            nlp1_cuda = torch.tensor(batched_data["nlp1"],dtype=torch.int64).to(device)
            nlp1_atten_mask = torch.tensor(batched_data["nlp1_atten_mask"],dtype=torch.int64).to(device)

            act_type = torch.tensor(batched_data["act_type"],dtype=torch.bool).to(device)
            #compute headtoken
            # headtoken = deepmodel_headtoken(s1_cuda,attention_mask=s1_atten_mask)
            # logits = deepmodel_a(headtoken,nlp1_cuda,act_type,attention_mask=nlp1_atten_mask)

        if iter>20:
            break

    for q_id in range(config_training.num_sampler):
        queue_in_list[q_id].put(("done", None, None))

    pywerewolf.utils.end_samplegame_group(proc_pool)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

