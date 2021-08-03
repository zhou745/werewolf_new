import numpy as np

class empty_headtoken_generator(object):

    def __init__(self):
        pass

    def __call__(self,system_state):
        return (None)