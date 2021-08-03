import os
def dumpy_current(save_dir,value_list):
    if not os.path.exists("log/" + save_dir+"/dump"):
        f_v = open("log/" + save_dir + "/dump", "w")
    else:
        f_v = open("log/" + save_dir + "/dump", "a")

    write_str = ""
    for v in value_list:
        write_str += str(v)[0:7] + " "
    f_v.writelines(write_str + "\n")
    f_v.close()

def create_files(save_dir):
    if not os.path.exists("ckpt/" + save_dir):
        os.mkdir("ckpt/" + save_dir)
    if not os.path.exists("log/" + save_dir):
        os.mkdir("log/" + save_dir)
    f_v = open("log/" + save_dir + "/value", "w")
    f_t = open("log/" + save_dir + "/target", "w")
    f_lv = open("log/" + save_dir + "/loss_v", "w")
    f_la = open("log/" + save_dir + "/loss_a", "w")
    f_prob = open("log/" + save_dir + "/prob", "w")
    f_stage = open("log/" + save_dir + "/stage", "w")
    f_v.close()
    f_t.close()
    f_lv.close()
    f_la.close()
    f_prob.close()
    f_stage.close()

def write_files(save_dir,critic_s1,estimate,loss_v,loss_a,state_stage):
    f_v = open("log/" + save_dir + "/value", "a")
    f_t = open("log/" + save_dir + "/target", "a")
    f_lv = open("log/" + save_dir + "/loss_v", "a")
    f_la = open("log/" + save_dir + "/loss_a", "a")
    f_stage = open("log/" + save_dir + "/stage", "a")
    v_list = critic_s1.view(-1).detach().cpu().numpy().tolist()
    t_list = estimate.view(-1).detach().cpu().numpy().tolist()
    lv = loss_v.detach().cpu().numpy().tolist()
    la = loss_a.detach().cpu().numpy().tolist()
    if isinstance(state_stage,list):
        stage = state_stage
    else:
        stage = state_stage.detach().cpu().numpy().tolist()

    write_str = ""
    for v in v_list:
        write_str += str(v)[0:7] + " "
    f_v.writelines(write_str + "\n")
    write_str = ""
    for v in t_list:
        write_str += str(v)[0:7] + " "
    f_t.writelines(write_str + "\n")
    write_str=""
    for s in stage:
        write_str += str(s) + " "
    f_stage.writelines(write_str + "\n")

    write_str = str(lv)
    f_lv.writelines(write_str + "\n")
    write_str = str(la)
    f_la.writelines(write_str + "\n")
    f_v.close()
    f_t.close()
    f_lv.close()
    f_la.close()
    f_stage.close()
