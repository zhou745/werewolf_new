class dict_headtoken_generator(object):

    def __init__(self,state_dict,offset_game):
        key_list = [key for key in state_dict.keys()]
        key_list.sort()
        self.key_list = key_list
        self.offset_game = offset_game

    def __call__(self,system_state):
        return (self.key_list.index(system_state)+self.offset_game)