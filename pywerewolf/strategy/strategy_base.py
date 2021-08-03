class strategy_base(object):
    def __init__(self,device,action_generator,q_generator,eta = 0.1,
                                                          epsilon = 0.05):
        self.action_generator = action_generator
        self.q_generator = q_generator
        self.device = device

        #sampling hyper paramter
        self.eta = eta    # probability for br
        self.epsilon = epsilon

    def action(self,system_state,nlp_state,action_mask,action_type):
        raise NotImplementedError("not implimented here")

    def set_eta(self,eta):
        self.eta = eta

    def set_epsilon(self,epsilon):
        self.epsilon = epsilon