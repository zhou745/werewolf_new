import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training_config')

#RL env settings
parser.add_argument('--game_per_sample', type=int, default=64)
parser.add_argument('--num_sampler', type=int, default=4)
parser.add_argument('--game_eta', type=int, default=0.1)
parser.add_argument('--game_epsilon', type=float, default=0.1)
parser.add_argument('--game_epsilon_scale', type=float, default=0.1)
parser.add_argument('--game_halfpoint',type=int, default=100000)
parser.add_argument('--game_temper',type=float,default=500.)

parser.add_argument('--num_player', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=16)
parser.add_argument('--game_compose', type=list, default=[0,0,1,7])
parser.add_argument('--game_manager', type=str, default="werewolf_manager_cyclic")
parser.add_argument('--headtoken_generator', type=str, default="bert_headtoken_generator")
parser.add_argument('--a_generator', type=str, default="bert_a_generator")
parser.add_argument('--q_generator', type=str, default="bert_q_generator")
parser.add_argument('--strategy', type=str, default="strategy_headtoken")
parser.add_argument('--memory_size',type=int,default=1024*1000)

#training schedule
parser.add_argument('--iterstart_memorysize_SL', type=int, default=128)
parser.add_argument('--iterstart_memorysize_RL', type=int, default=128)
parser.add_argument('--max_data_per_key', type=int, default=1024)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--use_ema_model', type=bool, default=False)
parser.add_argument('--batch_size_SL', type=int, default=1024)
parser.add_argument('--batch_size_RL', type=int, default=1024)

parser.add_argument('--actor_ema', type=float, default=1e-3)
parser.add_argument('--critic_q_ema', type=float, default=1e-2)
parser.add_argument('--loss_ema', type=float, default=1e-2)
parser.add_argument('--lr_act', type=float, default=1e-5)
parser.add_argument('--lr_q', type=float, default=1e-5)

parser.add_argument('--warmup', type=bool, default=False)
parser.add_argument('--warmup_iter', type=int, default=2000)
parser.add_argument('--lr_epsilon_scale', type=float, default=0.5)
parser.add_argument('--lr_halfpoint',type=int, default=100000)
parser.add_argument('--lr_temper',type=float,default=2000.)

parser.add_argument('--aligned_size', type=int, default=48)
parser.add_argument('--gamma', type=float, default=1.0)

#random settings
parser.add_argument('--seed', type=int, default=1234)

#save settings
parser.add_argument('--save_update', type=int, default=1000)
parser.add_argument('--max_update', type=int, default=2000000)

parser.add_argument('--save_dir', type=str, default="1w_1p_2v_cyclic_aug")
parser.add_argument('--max_grad_norm_q', type=float, default=7)
parser.add_argument('--max_grad_norm_a', type=float, default=7)
parser.add_argument('--normalize_loss', type=float, default=30)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--load_episode', type=int, default=0)

parser.add_argument('--state_list',  type=str,  default="1w_1p_2v_headtoken.npy")
parser.add_argument('--memory_bank',  type=str,  default="memory_1w_2v_1p_vocb_16.npy")

#model parameters
parser.add_argument('--hidden_size', type=int, default=96)
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_attention_heads', type=int, default=6)
parser.add_argument('--bert_intermediate_size', type=int, default=384)
parser.add_argument('--mlp_intermediate_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.)

def main(args):
    config_name = "1w_1p_2v_cyclic_aug_config"
    np.save("training_config/"+config_name,args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)