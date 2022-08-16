# -- coding: utf-8 --
# @Time : 2022/8/16
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box


import numpy as np


# def np_norm(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm

def np_norm(x):
    return (x - np.average(x)) / np.std(x)


class CalDistance:
    def __init__(self, vector1=None, vector2=None):
        if isinstance(vector1, list):
            self.vector1 = np.array(vector1)
        else:
            self.vector1 = vector1

        if isinstance(vector2, list):
            self.vector2 = np.array(vector2)
        else:
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


class Normalize:
    def __init__(self, aim_in):
        if isinstance(aim_in, list):
            self.aim_in = np.array(aim_in)
        else:
            self.aim_in = aim_in

    def np_norm(self):
        if len(self.aim_in) == 1:
            return self.aim_in
        return (self.aim_in - np.min(self.aim_in)) / (np.max(self.aim_in) - np.min(self.aim_in))
