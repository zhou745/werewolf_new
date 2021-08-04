import numpy as np

file_name = "rl_act_dict0.npy"

memory_list = np.load(file_name,allow_pickle=True).tolist()

a = 0
while a>-2:
    print(memory_list[a])
    a = int(input())