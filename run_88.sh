srun --partition=vc_research_2 --gres=gpu:0 -c16 -w SH-IDC1-10-198-8-128\
     --job-name=cyclic_game --kill-on-bad-exit=1\
      python train_gpu_list_memory.py --config_name 1w_1p_2v_cyclic_aug_config.npy