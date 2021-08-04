import numpy as np
import argparse
import pywerewolf
from tqdm import tqdm
import torch
import os
import socket
import time

from pywerewolf.werewolf_env.werewolf_manager_base import werewolf_manager_base
from pywerewolf.werewolf_env.werewolf_manager_cyclic import werewolf_manager_cyclic
from pywerewolf.werewolf_env.werewolf_manager_named import werewolf_manager_named
from pywerewolf.utils.tokenizer import tokenizer_base

from pywerewolf.strategy.dict_headtoken_generator import dict_headtoken_generator
from pywerewolf.strategy.bert_headtoken_generator import bert_headtoken_generator
from pywerewolf.strategy.bert_a_generator import bert_a_generator
from pywerewolf.strategy.bert_q_generator import bert_q_generator
from pywerewolf.strategy.strategy_headtoken import strategy_headtoken

parser = argparse.ArgumentParser(description='config_name')
parser.add_argument('--config_name', type=str)
parser.add_argument('--iter_to_load', type=int,default=2000)
parser.add_argument('--eval_type', type=str, default="win_rate")
parser.add_argument('--eval_num', type=int, default=1000)

def main(args):
    config_eval = np.load("training_config/" + args.config_name, allow_pickle=True).tolist()
    device = 0 if torch.cuda.is_available() else torch.device('cpu')
    #load deepmodels
    # config for the model
    manager = eval(config_eval.game_manager)(config_eval.num_player,
                                                      game_compose=config_eval.game_compose,
                                                      game_step = config_eval.game_step)
    tokenizer = tokenizer_base(manager.offset_special_token)

    config_model = pywerewolf.deepmodel.BertConfig(vocab_size=manager.offset_game_all,
                                                   hidden_size=config_eval.hidden_size,
                                                   num_hidden_layers=config_eval.num_hidden_layers,
                                                   num_attention_heads=config_eval.num_attention_heads,
                                                   intermediate_size=config_eval.bert_intermediate_size,
                                                   hidden_dropout_prob=config_eval.dropout,
                                                   attention_probs_dropout_prob=config_eval.dropout)

    # deepmodel_headtoken = None
    deepmodel_headtoken = pywerewolf.deepmodel.headtoken_model(config_model, config_eval.mlp_intermediate_size).to(device)
    deepmodel_a = pywerewolf.deepmodel.bert_headtoken_model(config_model, config_eval.num_player,
                                                            config_eval.vocab_size,
                                                            config_eval.mlp_intermediate_size).to(device)
    deepmodel_q = pywerewolf.deepmodel.bert_headtoken_model(config_model, config_eval.num_player,
                                                            config_eval.vocab_size,
                                                            config_eval.mlp_intermediate_size).to(device)

    #make strategys
    # create head token generator
    if config_eval.headtoken_generator.replace("_batch","") == "bert_headtoken_generator":
        headtoken_generator = bert_headtoken_generator(deepmodel_headtoken, tokenizer, device)
    else:
        raise RuntimeError("unknown headtoken generator type")

    # create a generator
    a_generator = eval(config_eval.a_generator.replace("_batch",""))(deepmodel_a, tokenizer, device)

    # create q generator
    q_generator = eval(config_eval.q_generator.replace("_batch",""))(deepmodel_q, tokenizer, device)

    # create strategy
    strategy = eval(config_eval.strategy.replace("_batch",""))(device, a_generator, q_generator, manager.offset_game_all, headtoken_generator,
                                   eta=0.,
                                   epsilon=config_eval.game_epsilon)

    if args.eval_type == "win_rate":
        record_game_list = []
        win_game = 0
        for idx in tqdm(range(args.eval_num)):
            manager.reset()

            status = "continue"
            actions = None
            one_game = []
            while status == "continue":
                sys_s, nlp_s, act_mask, act_type, reward, status, ct = manager.step(actions)
                # actions = rand_strategy(act_mask,act_type)
                # compute actions
                actions = []
                response = []
                policy_list = []
                q_list = []
                for i in range(manager.num_player):
                    act, rep,policy,q_value = strategy.eval(sys_s[i], nlp_s[i], act_mask, act_type[i])
                    actions.append(act)
                    response.append(rep)
                    policy_list.append(policy)
                    q_list.append(q_value)
                one_game.append([sys_s, nlp_s, act_mask, act_type, reward, status, ct, actions, response,policy_list,q_list])
            if "were" in status:
                win_game +=1

            record_game_list.append(one_game)
        if not os.path.exists("eval/" + config_eval.save_dir):
            os.mkdir("eval/" + config_eval.save_dir)
        np.save("eval/" + config_eval.save_dir+"/game_record"+str(args.iter_to_load),record_game_list)
        print("win_rate of werewolf is %f"%(win_game/args.eval_num))
    else:
        raise RuntimeError("unknown eval type")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)