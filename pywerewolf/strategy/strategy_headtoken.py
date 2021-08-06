from pywerewolf.strategy.strategy_base import strategy_base
import random
import numpy as np

#process one agent one action at a time
class strategy_headtoken(strategy_base):
    def __init__(self,device,action_generator,q_generator,game_token_num,headtoken_generator,eta = None,
                                                                                             epsilon = None):

        hyper_para = {}
        if eta != None:
            hyper_para.update({"eta": eta})
        if epsilon != None:
            hyper_para.update({"epsilon": epsilon})

        super(strategy_headtoken,self).__init__(device,action_generator,q_generator,**hyper_para)

        self.offset_game = game_token_num
        #dict to store the headtoken states
        self.headtoken_generator = headtoken_generator


    def action(self,system_state,nlp_state,action_mask,action_type):
        if action_type=="none":
            return(0,"ave")

        head_token = self.headtoken_generator(system_state)
        #decide whether average policy or BR is used
        policy_choose = random.uniform(0.,1)


        #BR is used
        if policy_choose<self.eta:
            q_value = self.q_generator(head_token,nlp_state,action_mask,action_type)

            num_action = q_value.shape[0]
            prob = [self.epsilon/num_action for idx in range(num_action)]
            max_idx = np.argmax(q_value)
            prob[max_idx] += 1- self.epsilon

            a_policy = np.array(prob,dtype=float)
            Response_type = "br"
        else:
            a_policy = self.action_generator(head_token,nlp_state,action_mask,action_type)
            Response_type = "ave"

        #normalize it to ensure sum < 1
        a_policy = a_policy / np.sum(a_policy + 1e-7)

        target_act = self.select_action(a_policy,action_mask)

        return (target_act,Response_type)

    def eval(self,system_state,nlp_state,action_mask,action_type):
        if action_type=="none":
            return(0,"ave",None,None)

        head_token = self.headtoken_generator(system_state)
        #decide whether average policy or BR is used
        policy_choose = random.uniform(0.,1)
        #compute q value
        q_value = self.q_generator(head_token,nlp_state,action_mask,action_type)
        #compute policy
        a_policy = self.action_generator(head_token,nlp_state,action_mask,action_type)
        Response_type = "ave"

        #normalize it to ensure sum < 1
        a_policy = a_policy / np.sum(a_policy + 1e-7)

        target_act = self.select_action(a_policy,action_mask)

        return (target_act,Response_type,a_policy,q_value)

    def select_action(self,a_policy,action_mask):

        if action_mask == None:
            choice_list = np.random.multinomial(1, a_policy, 1)[0]
            choice_idx = np.argmax(choice_list)
        else:
            musked = -1
            while musked<-0.5:
                choice_list = np.random.multinomial(1, a_policy, 1)[0]
                choice_idx = np.argmax(choice_list)
                musked = action_mask[choice_idx]
        return (choice_idx)


