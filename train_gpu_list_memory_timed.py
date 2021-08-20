import numpy as np
import argparse
import pywerewolf
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
import torch
import os
import socket
import time

from torch.utils.tensorboard import SummaryWriter

from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base
from pywerewolf.werewolf_env.werewolf_manager_cyclic import werewolf_manager_cyclic
from pywerewolf.werewolf_env.werewolf_manager_named import werewolf_manager_named
from pywerewolf.werewolf_env.werewolf_manager_timed import werewolf_manager_timed
from pywerewolf.werewolf_env.werewolf_manager_timed_cyclic import werewolf_manager_timed_cyclic

parser = argparse.ArgumentParser(description='config_name')
parser.add_argument('--config_name', type=str)


# os.environ['CUDA_VISIBLE_DEVICES']="1,2,3,4"

def main(args):
    pid = int(os.environ["SLURM_PROCID"])
    jobid = os.environ["SLURM_JOBID"]

    config_training = np.load("training_config/" + args.config_name, allow_pickle=True).tolist()

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

    mp.spawn(model_parallel, args=(pid, dist_url, config_training),
             nprocs=config_training.world_size,
             join=True)


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def setup(rank, dist_url, world_size, name='nccl'):
    dist.init_process_group(name, init_method=dist_url, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def sig_decay(iter, delta, half_point, low_bound, Temperature):
    return (delta / (1 + np.exp((iter - half_point) / Temperature)) + low_bound)


def model_parallel(rank, pid, dist_url, config_training):
    gpu = rank
    rank = pid * 8 + gpu
    # set up the current training env
    device = gpu if torch.cuda.is_available() else torch.device('cpu')
    print("current rank is %d" % (rank), flush=True)
    setup(rank, dist_url, config_training.world_size)

    if not os.path.exists("tensorboard/" + config_training.save_dir):
        os.mkdir("tensorboard/" + config_training.save_dir)

    if device == 0:
        writer = SummaryWriter("tensorboard/" + config_training.save_dir)

    # config for the model
    manager_fake = eval(config_training.game_manager)(config_training.num_player,
                                                      game_compose=config_training.game_compose)
    tokenizer = pywerewolf.utils.tokenizer_base(manager_fake.offset_special_token)

    key_dict = np.load("headtoken_dict/"+config_training.key_dict,allow_pickle=True).tolist()

    config_model = pywerewolf.deepmodel.BertConfig(vocab_size=manager_fake.offset_game_all+len(key_dict.keys()),
                                                   hidden_size=config_training.hidden_size,
                                                   num_hidden_layers=config_training.num_hidden_layers,
                                                   num_attention_heads=config_training.num_attention_heads,
                                                   intermediate_size=config_training.bert_intermediate_size,
                                                   hidden_dropout_prob=config_training.dropout,
                                                   attention_probs_dropout_prob=config_training.dropout)

    # deepmodel_headtoken = None
    # deepmodel_a = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,config_training.vocab_size)
    # deepmodel_q = pywerewolf.deepmodel.bert_headtoken_model(config_model,config_training.num_player,config_training.vocab_size)

    # deepmodel_headtoken = pywerewolf.deepmodel.headtoken_model(config_model, config_training.mlp_intermediate_size).to(
    #     device)
    # deepmodel_a = pywerewolf.deepmodel.bert_headtoken_model(config_model, config_training.num_player,
    #                                                         config_training.vocab_size,
    #                                                         config_training.mlp_intermediate_size).to(device)
    # deepmodel_q = pywerewolf.deepmodel.bert_headtoken_model(config_model, config_training.num_player,
    #                                                         config_training.vocab_size,
    #                                                         config_training.mlp_intermediate_size).to(device)

    deepmodel_a = pywerewolf.deepmodel.dict_timed_headtoken_model(config_model, config_training.num_player,
                                                            config_training.vocab_size,
                                                            config_training.mlp_intermediate_size,
                                                            max_time=config_training.max_time).to(device)
    deepmodel_q = pywerewolf.deepmodel.dict_timed_headtoken_model(config_model, config_training.num_player,
                                                            config_training.vocab_size,
                                                            config_training.mlp_intermediate_size,
                                                            max_time=config_training.max_time).to(device)
    deepmodel_headtoken = None
    # deepmodel_headtoken.share_memory()
    deepmodel_a.share_memory()
    deepmodel_q.share_memory()

    queue_in_list = [mp.Queue(1) for i in range(config_training.num_sampler)]
    queue_out_list = [mp.Queue(1) for i in range(config_training.num_sampler)]

    # proc_pool = pywerewolf.utils.start_samplegame_group(deepmodel_headtoken,deepmodel_a,deepmodel_q,
    #                                                     queue_in_list, queue_out_list,device,config_training)

    proc_pool = pywerewolf.utils.start_samplegame_group_batch(deepmodel_headtoken, deepmodel_a, deepmodel_q,
                                                              queue_in_list, queue_out_list, device, config_training)

    # deepmodel_headtoken_ddp = DDP(deepmodel_headtoken, device_ids=[device])
    deepmodel_q_ddp = DDP(deepmodel_q, device_ids=[device])
    deepmodel_a_ddp = DDP(deepmodel_a, device_ids=[device])

    # syn the initial paramters of global model
    # for parameter in deepmodel_headtoken_ddp.parameters():
    #     torch.distributed.broadcast(parameter, 0, async_op=False)

    for parameter in deepmodel_q_ddp.parameters():
        torch.distributed.broadcast(parameter, 0, async_op=False)

    for parameter in deepmodel_a_ddp.parameters():
        torch.distributed.broadcast(parameter, 0, async_op=False)

    # memory dict and list
    # act_state_dict = {}
    # statement_state_dict = {}
    # act_state_dict_sl = {}
    # statement_state_dict_sl = {}

    act_state_dict = [None for i in range(config_training.memory_size)]
    statement_state_dict = [None for i in range(config_training.memory_size)]
    act_state_dict_sl = [None for i in range(config_training.memory_size)]
    statement_state_dict_sl = [None for i in range(config_training.memory_size)]

    # data store
    # store_data = pywerewolf.utils.store_data_base_fast(1024)
    store_data = pywerewolf.utils.list_store_data_fast(config_training.memory_size)

    sampler_rl = pywerewolf.utils.uniform_sampler(config_training.batch_size_RL)
    sampler_sl = pywerewolf.utils.uniform_sampler(config_training.batch_size_SL)

    # augment = pywerewolf.utils.cyclic_player(manager_fake.offset_position,
    #                                          manager_fake.offset_career,
    #                                          manager_fake.offset_vocb,
    #                                          manager_fake.offset_special_token,
    #                                          manager_fake.offset_relative_pos,
    #                                          manager_fake.num_player)
    augment = pywerewolf.utils.cyclic_timed_player(manager_fake.offset_position,
                                             manager_fake.offset_career,
                                             manager_fake.offset_vocb,
                                             manager_fake.offset_special_token,
                                             manager_fake.offset_relative_pos,
                                             manager_fake.num_player)

    batch_collect = pywerewolf.utils.batch_timed_collector(tokenizer, config_training.aligned_size,
                                                                config_training.num_player,
                                                                key_dict,manager_fake.offset_game_all)

    # optimizers
    # optimizer_headtoken = optim.Adam(deepmodel_headtoken_ddp.parameters(), betas=(0.9, 0.999),
    #                                  lr=config_training.lr_act, eps=1e-6, weight_decay=0., amsgrad=False)

    optimizer_a = optim.Adam(deepmodel_a_ddp.parameters(), betas=(0.9, 0.999), lr=config_training.lr_act, eps=1e-6,
                             weight_decay=0., amsgrad=False)

    optimizer_q = optim.Adam(deepmodel_q_ddp.parameters(), betas=(0.9, 0.999), lr=config_training.lr_q, eps=1e-6,
                             weight_decay=0., amsgrad=False)

    # main training loop
    for iter in range(config_training.max_update):
        memory = []
        # sampling once
        for q_id in range(config_training.num_sampler):
            queue_in_list[q_id].put(("sample", None, None))
        for q_id in range(config_training.num_sampler):
            memory += queue_out_list[q_id].get()

        # for q_id in range(config_training.num_sampler):
        #     queue_in_list[q_id].put(("done", None, None))

        # pywerewolf.utils.end_samplegame_group(proc_pool)
        # process game data
        if device == 0:
            str_loop = "tqdm(memory)"
        else:
            str_loop = "memory"
        for game in eval(str_loop):
            store_data(act_state_dict, statement_state_dict, act_state_dict_sl, statement_state_dict_sl, game)

        deepmodel_q_ddp.train()
        deepmodel_q_ddp.train()
        # deepmodel_headtoken_ddp.train()

        # loss
        loss_a_mean = 0.
        loss_q_mean = 0.

        loss_a_act_mean = 0.
        loss_a_statement_mean = 0.

        loss_q_act_mean = 0.
        loss_q_statement_mean = 0.

        # sample data for training
        if store_data.statement_state_total > config_training.iterstart_memorysize_RL and \
                store_data.act_state_total > config_training.iterstart_memorysize_RL:
            if device == 0:
                loop_str = "tqdm(range(config_training.iter_RL))"
            else:
                loop_str = "range(config_training.iter_RL)"

            for iter_rl in eval(loop_str):
                # reset train
                loss_q = 0.
                sampled_act = sampler_rl(act_state_dict, store_data.act_state_total)
                sampled_statement = sampler_rl(statement_state_dict, store_data.statement_state_total)

                sampled_data = sampled_act + sampled_statement
                # augment data act
                augmented_sample_data = []
                for one_action in sampled_data:
                    one_action_aug = augment(one_action)
                    augmented_sample_data.append(one_action_aug)

                # collect batch
                batched_data = batch_collect(augmented_sample_data)

                # conver to cuda
                s1_cuda = torch.tensor(batched_data["s1"], dtype=torch.int64).to(device)
                # s1_atten_mask = torch.tensor(batched_data["s1_atten_mask"], dtype=torch.int64).to(device)

                nlp1_cuda = torch.tensor(batched_data["nlp1"], dtype=torch.int64).to(device)
                nlp1_atten_mask = torch.tensor(batched_data["nlp1_atten_mask"], dtype=torch.int64).to(device)

                act_type = torch.tensor(batched_data["act_type"], dtype=torch.bool).to(device)
                # compute headtoken
                q_value = deepmodel_q_ddp(s1_cuda, nlp1_cuda, act_type, attention_mask=nlp1_atten_mask)

                estimate = torch.tensor(batched_data["finnal_reward"], dtype=torch.float32).to(device)
                action_ids = torch.tensor(batched_data["act_ids"], dtype=torch.int64).to(device)

                # compute loss
                target_act = estimate[act_type]
                target_nlp = estimate[~act_type]

                index_act = action_ids[act_type].view(-1, 1)
                index_nlp = action_ids[~act_type].view(-1, 1)

                pred_q_act = q_value["act"].gather(1, index_act)
                pred_q_nlp = q_value["statement"].gather(1, index_nlp)

                loss_q_act = (0.5 * (pred_q_act.view(-1) - target_act) ** 2).mean()
                loss_q_nlp = (0.5 * (pred_q_nlp.view(-1) - target_nlp) ** 2).mean()

                loss_q = loss_q_act + loss_q_nlp + \
                         (q_value["act"] * q_value["act"]).sum() * 0. + (
                                     q_value["statement"] * q_value["statement"]).sum() * 0.

                # update parameters
                # optimizer_headtoken.zero_grad()
                optimizer_q.zero_grad()

                loss_q.backward()

                # optimizer_headtoken.step()
                optimizer_q.step()

                if device == 0:
                    loss_q_mean = loss_q.detach().cpu().numpy() * config_training.loss_ema + loss_q_mean * (
                                1 - config_training.loss_ema)
                    loss_q_act_mean = loss_q_act.detach().cpu().numpy() * config_training.loss_ema + loss_q_act_mean * (
                                1 - config_training.loss_ema)
                    loss_q_statement_mean = loss_q_nlp.detach().cpu().numpy() * config_training.loss_ema + loss_q_statement_mean * (
                                1 - config_training.loss_ema)

        # train action
        if store_data.statement_sl_state_total > config_training.iterstart_memorysize_SL and \
                store_data.act_state_sl_total > config_training.iterstart_memorysize_SL:

            if device == 0:
                loop_str = "tqdm(range(config_training.iter_SL))"
            else:
                loop_str = "range(config_training.iter_SL)"

            for iter_sl in eval(loop_str):
                sampled_act = sampler_sl(act_state_dict_sl, store_data.act_state_sl_total)
                sampled_statement = sampler_sl(statement_state_dict_sl, store_data.statement_sl_state_total)

                sampled_data = sampled_act + sampled_statement
                # augment data act
                augmented_sample_data = []
                for one_action in sampled_data:
                    one_action_aug = augment(one_action)
                    augmented_sample_data.append(one_action_aug)

                # collect batch
                batched_data = batch_collect(augmented_sample_data)

                # conver to cuda
                s1_cuda = torch.tensor(batched_data["s1"], dtype=torch.int64).to(device)
                # s1_atten_mask = torch.tensor(batched_data["s1_atten_mask"], dtype=torch.int64).to(device)

                nlp1_cuda = torch.tensor(batched_data["nlp1"], dtype=torch.int64).to(device)
                nlp1_atten_mask = torch.tensor(batched_data["nlp1_atten_mask"], dtype=torch.int64).to(device)

                act_type = torch.tensor(batched_data["act_type"], dtype=torch.bool).to(device)
                # compute headtoken
                # headtoken = deepmodel_headtoken_ddp(s1_cuda, attention_mask=s1_atten_mask)
                logits = deepmodel_a_ddp(s1_cuda, nlp1_cuda, act_type, attention_mask=nlp1_atten_mask)

                action_ids = torch.tensor(batched_data["act_ids"], dtype=torch.int64).to(device)
                action_mask = torch.tensor(batched_data["act_mask"], dtype=torch.float32).to(device)
                # comnpute loss
                gather_index_act = action_ids[act_type].view(-1, 1)
                gather_index_nlp = action_ids[~act_type].view(-1, 1)
                act_mask_tensor = action_mask[act_type]

                prob_act = F.softmax(logits["act"] + act_mask_tensor, dim=-1)
                prob_nlp = F.softmax(logits["statement"], dim=-1)

                prob_act_selected = prob_act.gather(1, gather_index_act)
                prob_nlp_selected = prob_nlp.gather(1, gather_index_nlp)

                # log loss for act

                loss_act = -torch.log(prob_act_selected).mean()
                loss_nlp = -torch.log(prob_nlp_selected).mean()
                loss_a = loss_act + loss_nlp + (prob_act * prob_act).sum() * 0. + (prob_nlp * prob_nlp).sum() * 0.

                # update policy parameters
                optimizer_a.zero_grad()
                # optimizer_headtoken.zero_grad()
                loss_a.backward()
                # compute the current norm of gradient
                # torch.nn.utils.clip_grad_norm_(actor_ddp.parameters(), args.max_grad_norm_a)
                optimizer_a.step()
                # optimizer_headtoken.step()

                if device == 0:
                    loss_a_mean = loss_a.detach().cpu().numpy() * config_training.loss_ema + loss_a_mean * (
                                1 - config_training.loss_ema)
                    loss_a_act_mean = loss_act.detach().cpu().numpy() * config_training.loss_ema + loss_a_act_mean * (
                                1 - config_training.loss_ema)
                    loss_a_statement_mean = loss_nlp.detach().cpu().numpy() * config_training.loss_ema + loss_a_statement_mean * (
                                1 - config_training.loss_ema)

        # print loss and save loss
        if device == 0:
            print("at iter %d current loss is %f %f" % (iter,loss_q_mean, loss_a_mean),flush=True)

            writer.add_scalar('Loss/a', loss_a_mean, iter)
            writer.add_scalar('Loss/q', loss_q_mean, iter)
            writer.add_scalar('Loss_a/act', loss_a_act_mean, iter)
            writer.add_scalar('Loss_a/statement', loss_a_statement_mean, iter)
            writer.add_scalar('Loss_q/act', loss_q_act_mean, iter)
            writer.add_scalar('Loss_q/statement', loss_q_statement_mean, iter)

        # decay lr
        for g in optimizer_a.param_groups:
            g['lr'] = sig_decay(iter, (1 - config_training.lr_epsilon_scale) * config_training.lr_q,
                                config_training.lr_halfpoint,
                                config_training.lr_epsilon_scale * config_training.lr_q,
                                config_training.lr_temper)

        for g in optimizer_q.param_groups:
            g['lr'] = sig_decay(iter, (1 - config_training.lr_epsilon_scale) * config_training.lr_q,
                                config_training.lr_halfpoint,
                                config_training.lr_epsilon_scale * config_training.lr_q,
                                config_training.lr_temper)

        # for g in optimizer_headtoken.param_groups:
        #     g['lr'] = sig_decay(iter, (1 - config_training.lr_epsilon_scale) * config_training.lr_q,
        #                         config_training.lr_halfpoint,
        #                         config_training.lr_epsilon_scale * config_training.lr_q,
        #                         config_training.lr_temper)

        # save ckpt memory
        if iter % config_training.save_update == 0 and device == 0:
            if not os.path.exists("ckpt/" + config_training.save_dir):
                os.mkdir("ckpt/" + config_training.save_dir)
            torch.save(deepmodel_a_ddp.module.state_dict(),
                       "ckpt/" + config_training.save_dir + "/act_update" + str(iter))
            torch.save(deepmodel_q_ddp.module.state_dict(),
                       "ckpt/" + config_training.save_dir + "/q_update" + str(iter))
            torch.save(optimizer_a.state_dict(),
                       "ckpt/" + config_training.save_dir + "/a_optim" + str(iter))
            torch.save(optimizer_q.state_dict(),
                       "ckpt/" + config_training.save_dir + "/q_optim" + str(iter))
            # torch.save(deepmodel_headtoken_ddp.module.state_dict(),
            #            "ckpt/" + config_training.save_dir + "/headtoken_update" + str(iter))
            # dump the current memory
            if not os.path.exists("memory/" + config_training.save_dir):
                os.mkdir("memory/" + config_training.save_dir)
            np.save("memory/" + config_training.save_dir + "/sl_act_dict" + str(iter), act_state_dict_sl)
            np.save("memory/" + config_training.save_dir + "/sl_nlp_dict" + str(iter), statement_state_dict_sl)
            np.save("memory/" + config_training.save_dir + "/rl_act_dict" + str(iter), act_state_dict)
            np.save("memory/" + config_training.save_dir + "/rl_nlp_dict" + str(iter), statement_state_dict)

    if device == 0:
        writer.close()

    for q_id in range(config_training.num_sampler):
        queue_in_list[q_id].put(("done", None, None))

    # pywerewolf.utils.end_samplegame_group(proc_pool)
    pywerewolf.utils.end_samplegame_group_batch(proc_pool)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

