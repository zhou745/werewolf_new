srun --partition=vc_research_2 --gres=gpu:4\
     --job-name=cyclic_game --kill-on-bad-exit=1\
      python train_gpu_list_memory_timed.py --config_name 2w_1g_1p_2v_timed_vocb64_cyclic_aug_batch.npy