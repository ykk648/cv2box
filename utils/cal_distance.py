import numpy as np
from .util import np_norm


class CalDistance:
    def __init__(self, vector1, vector2):
        self.vector1 = vector1
        self.vector2 = vector2

    def euc(self):
        return np.sqrt(np.sum(np.square(self.vector1 - self.vector2)))

    def euc_2(self):
        return np.linalg.norm(self.vector1 - self.vector2)

    def euc_norm(self):
        return np.linalg.norm(np_norm(self.vector1) - (self.vector2))
