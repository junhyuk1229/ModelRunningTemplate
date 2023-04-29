import numpy as np


class MatrixFactorization:
    def __init__(self, settings):
        self.P = np.random.normal(size=(settings["user_num"], settings["mf"]["k"]))
        self.Q = np.random.normal(size=(settings["book_num"], settings["mf"]["k"]))
        
        self.b = np.zeros(1)
        self.b_u = np.zeros(settings["user_num"])
        self.b_i = np.zeros(settings["book_num"])

    def train(self):
        pass