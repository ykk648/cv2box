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

    @staticmethod
    def sim(vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def cosine_ne(array1, array2):
        import numexpr as ne
        # fast face similarity cal func
        sumyy = np.einsum('ij,ij->i', array2, array2)
        sumxx = np.einsum('ij,ij->i', array1, array1)[:, None]
        sumxy = array1.dot(array2.T)
        sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
        sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
        return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')
