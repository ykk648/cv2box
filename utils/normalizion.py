# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import numpy as np


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
