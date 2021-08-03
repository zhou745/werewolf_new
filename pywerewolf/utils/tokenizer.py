default_special_token_list = ["game_start","check_werewolf_team","werewolf_agreed_kill",
                              "prophet_check_result","bad","good",
                              "dead_people","plyer_last_statement","player_statement",
                              "none","statement","collected","voted_out"]

class tokenizer_base(object):
    def __init__(self,offset_special_token,special_token_list = default_special_token_list):

        self.offset_special_token = offset_special_token
        self.special_token_list = special_token_list

    def __call__(self,nlp_state):
        nlp_state_list = nlp_state.split(" ")
        nlp_state_index_list = []
        for token in nlp_state_list:
            if token in self.special_token_list:
                index = self.special_token_list.index(token)+self.offset_special_token
            else:
                index = int(token)

            assert isinstance(index,int)
            nlp_state_index_list.append(index)
        return(nlp_state_index_list)