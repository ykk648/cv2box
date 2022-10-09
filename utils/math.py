# -- coding: utf-8 --
# @Time : 2022/8/16
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box


import numpy as np
from .util import try_import


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

    # def euc_norm(self):
    #     return np.linalg.norm(np_norm(self.vector1) - (self.vector2))

    @staticmethod
    def sim(vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def cosine_ne(array1, array2):
        ne = try_import('numexpr', 'cv_math: need numexpr here.')
        # fast face similarity cal func
        sumyy = np.einsum('ij,ij->i', array2, array2)
        sumxx = np.einsum('ij,ij->i', array1, array1)[:, None]
        sumxy = array1.dot(array2.T)
        sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
        sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
        return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')


class Normalize:
    """
    when process embedding like [1,512], np_norm=torch_l2_norm=torch_f_norm， but different in higher dimension
    """

    def __init__(self, aim_in):
        if isinstance(aim_in, list):
            self.aim_in = np.array(aim_in)
        else:
            self.aim_in = aim_in

    def np_max_min_norm(self):
        if len(self.aim_in) == 1:
            return self.aim_in
        return (self.aim_in - np.min(self.aim_in)) / (np.max(self.aim_in) - np.min(self.aim_in))

    def np_std_norm(self):
        return (self.aim_in - np.average(self.aim_in)) / np.std(self.aim_in)

    def np_norm(self):
        """
        same as:
            from sklearn.preprocessing import normalize
            normalize(x[:,np.newaxis], axis=0).ravel()
        :return:
        """
        norm = np.linalg.norm(self.aim_in)
        if norm == 0:
            return self.aim_in
        return self.aim_in / norm

    def torch_l2_norm(self, axis=1):
        torch = try_import('torch', 'cv_math: need torch here.')
        if not isinstance(self.aim_in, torch.tensor):
            self.aim_in = torch.tensor(self.aim_in)
        norm = torch.norm(self.aim_in, 2, axis, True)
        output = torch.div(self.aim_in, norm)
        return output

    def torch_f_norm(self):
        torch = try_import('torch', 'cv_math: need torch here.')
        if not isinstance(self.aim_in, torch.tensor):
            self.aim_in = torch.tensor(self.aim_in)
        return torch.nn.functional.normalize(torch.tensor(self.aim_in), p=2, dim=1)
