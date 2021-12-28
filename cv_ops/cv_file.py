# -- coding: utf-8 --
# @Time : 2021/12/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import pickle
from pathlib import Path
import json
import numpy as np


class CVFile:
    def __init__(self, file_path, file_format=None):
        self.file_data = None
        self.file_path = file_path

        if Path(self.file_path).exists():
            suffix = Path(self.file_path).suffix
            if suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    self.file_data = pickle.load(f)
            elif suffix == '.txt':
                with open(file_path, 'rb') as f:
                    self.file_data = f.readlines()
            elif suffix == '.json':
                with open(file_path, 'rb') as f:
                    self.file_data = json.load(f)

    @property
    def data(self):
        return self.file_data

    def pickle_write(self, data_in):
        with open(self.file_path, 'w') as f:
            pickle.dump(data_in, f)

    def json_write(self, data_in):
        for k, v in data_in.items():
            if isinstance(v, np.bool_):
                data_in[k] = bool(v)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_in, f)


if __name__ == '__main__':
    # pkl = ''
    # print(CVFile(json_path).data)

    json_path = '/home/tyler/图片/BingWallpaper/20210810-DinoShower_ZH-CN1791773864_UHD.json'
    print(CVFile(json_path).data['flags']['1'])
