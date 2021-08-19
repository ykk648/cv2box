import numpy as np


class CalDistance:
    def __init__(self, vector1, vector2):
        self.vector1 = vector1
        self.vector2 = vector2

    def euc(self):
        return np.sqrt(np.sum(np.square(self.vector1 - self.vector2)))
