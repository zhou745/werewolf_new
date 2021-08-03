from pywerewolf.strategy.strategy_base import strategy_base
import numpy as np
import torch

#process one agent one action at a time
class strategy_headtoken_batch(strategy_base):
    def __init__(self,device,action_generator,q_generator,game_token_num,headtoken_generator,eta = None,
                                                                                             epsilon = None):

        hyper_para = {}
        if eta != None:
            hyper_para.update({"eta": eta})
        if epsilon != None:
            hyper_para.update({"epsilon": epsilon})

        super(strategy_headtoken_batch,self).__init__(device,action_generator,q_generator,**hyper_para)

        self.offset_game = game_token_num
        #dict to store the headtoken states
        self.headtoken_generator = headtoken_generator



    def action(self,system_state_b,nlp_state_b,action_mask_b,action_type_b):
        #make sure none is not in action_type_b
        assert "none" not in action_type_b
        b = len(action_type_b)

        #fisrt seperate

        head_token = self.headtoken_generator(system_state_b)
        #decide whether average policy or BR is used
        policy_bool = np.random.uniform(low=0., high=1., size=(b,))<self.eta
        policy_q_index = np.where(policy_bool == True)[0]
        policy_a_index = np.where(policy_bool == False)[0]

        #compute q policy
        q_value = self.q_generator(head_token[policy_bool],nlp_state_b[policy_bool],action_mask_b[policy_bool],action_type_b[policy_bool])
        q_act_index = np.where(action_type_b[policy_bool]=="act")[0]
        q_statement_index = np.where(action_type_b[policy_bool]!="act")[0]

        #compute a policy
        a_policy = self.action_generator(head_token[~policy_bool],nlp_state_b[~policy_bool],action_mask_b[~policy_bool],action_type_b[~policy_bool])
        a_act_index = np.where(action_type_b[~policy_bool]=="act")[0]
        a_statement_index = np.where(action_type_b[~policy_bool]!="act")[0]

        
        #restore q and a back in order
        value_list = [None for i in range(system_state_b.shape[0])]
        response_list = [None for i in range(system_state_b.shape[0])]

        index_list = [i for i in range(system_state_b.shape[0])]

        #process q act
        if len(q_value["act"])>0:
            for idx in range(q_value["act"].shape[0]):
                q_group_idx  = q_act_index[idx]
                origin_idx = policy_q_index[q_group_idx]
                #compute a policy
                num_action = q_value["act"][idx].shape[0]
                prob = [self.epsilon / num_action for idx in range(num_action)]
                max_idx = np.argmax(q_value["act"][idx])
                prob[max_idx] += 1 - self.epsilon

                policy = np.array(prob, dtype=float)
                policy = policy / np.sum(policy + 1e-7)
                value_list[origin_idx] = policy
                response_list[origin_idx] = "br"

                assert origin_idx in index_list
                index_list.remove(origin_idx)

        # process q statement
        if len(q_value["statement"]):
            for idx in range(q_value["statement"].shape[0]):
                q_group_idx = q_statement_index[idx]
                origin_idx = policy_q_index[q_group_idx]
                #compute a policy
                num_action = q_value["statement"][idx].shape[0]
                prob = [self.epsilon / num_action for idx in range(num_action)]
                max_idx = np.argmax(q_value["statement"][idx])
                prob[max_idx] += 1 - self.epsilon

                policy = np.array(prob, dtype=float)
                policy = policy / np.sum(policy + 1e-7)
                value_list[origin_idx] = policy
                response_list[origin_idx] = "br"

                assert origin_idx in index_list
                index_list.remove(origin_idx)

        # process q act
        if len(a_policy["act"])>0:
            for idx in range(a_policy["act"].shape[0]):
                a_group_idx = a_act_index[idx]
                origin_idx = policy_a_index[a_group_idx]

                policy = a_policy["act"][idx]
                policy = policy / np.sum(policy + 1e-7)
                value_list[origin_idx] = policy
                response_list[origin_idx] = "br"

                assert origin_idx in index_list
                index_list.remove(origin_idx)

        # process a statement
        if len(a_policy["statement"])>0:
            for idx in range(a_policy["statement"].shape[0]):
                a_group_idx = a_statement_index[idx]
                origin_idx = policy_a_index[a_group_idx]

                policy = a_policy["statement"][idx]
                policy = policy / np.sum(policy + 1e-7)
                value_list[origin_idx] = policy
                response_list[origin_idx] = "br"

                assert origin_idx in index_list
                index_list.remove(origin_idx)

        #compute action list
        action_list = []
        for action_idx in range(system_state_b.shape[0]):
            action_list.append(self.select_action(value_list[action_idx], action_mask_b[action_idx]))
        return (action_list,response_list)

    def select_action(self,a_policy,action_mask):
        sum = action_mask.sum()

        if sum >2:
            choice_list = np.random.multinomial(1, a_policy, 1)[0]
            choice_idx = np.argmax(choice_list)
        else:
            musked = -1
            while musked<-0.5:
                choice_list = np.random.multinomial(1, a_policy, 1)[0]
                choice_idx = np.argmax(choice_list)
                musked = action_mask[choice_idx]
        return (choice_idx)


