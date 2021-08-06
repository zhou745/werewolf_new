import numpy as np

# file_name = "eval/2w_1g_1p_2v_cyclic_aug_batch_exp2/game_record12000.npy"
file_name = "eval/2w_1g_1p_2v_timed_cyclic_aug_batch/game_record9000.npy"

game_list = np.load(file_name,allow_pickle=True).tolist()

game_id = 0
while game_id>-2:
    game = game_list[game_id]
    for game_step in game:
        for idx in range(len(game_step)):
            print(game_step[idx])
        print("----------------------------------------------------")

    game_id = int(input())