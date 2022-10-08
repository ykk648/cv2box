# -- coding: utf-8 --
# @Time : 2021/12/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import pickle
from pathlib import Path
import json
import numpy as np
import os
import base64


def data_resolve(data_in, iter_times, dummy):
    """
    不同结构的循环解释器
    :param data_in:
    :param iter_times:
    :param dummy:
    :return:
    """
    if iter_times == 0:
        return
    blank_space = ' ' * (dummy - iter_times)
    iter_times -= 1
    data_type = type(data_in)

    if data_type is dict:
        print('{}dict keys: {}'.format(blank_space, data_in.keys()))
        for k, v in data_in.items():
            print('{}key {}, type {}'.format(blank_space, k, type(v)))
            data_resolve(v, iter_times, dummy)
    elif data_type is list:
        print('{}list length: {}, list head: {}'.format(blank_space, len(data_in), data_in[0]))
        data_resolve(data_in[0], iter_times, dummy)
    else:
        print('{}data: {}'.format(blank_space, data_in))


class CVFile:
    def __init__(self, file_path, *args, **kwargs):
        self.file_data = None
        self.file_path = file_path

        if Path(self.file_path).exists():
            self.suffix = Path(self.file_path).suffix
            if self.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    self.file_data = pickle.load(f)
            elif self.suffix == '.txt':
                with open(file_path, 'rb') as f:
                    self.file_data = f.readlines()
            elif self.suffix == '.json':
                with open(file_path, 'rb') as f:
                    self.file_data = json.load(f)
            elif self.suffix == '.npz':
                # h = arrays['key'][()]
                self.file_data = np.load(file_path, allow_pickle=True)
            elif self.suffix == '.h5':
                import h5py
                self.file_data = h5py.File(self.file_path, "r")
            elif self.suffix == '.npy':
                self.file_data = np.load(self.file_path, allow_pickle=True)
            elif self.suffix == '.yaml' or self.suffix == '.yml':
                import yaml
                with open(file_path, 'rb') as f:
                    self.file_data = yaml.safe_load(f)
            elif self.suffix == '.csv':
                import pandas as pd
                self.file_data = pd.read_csv(file_path, **kwargs)

    @property
    def data(self):
        return self.file_data

    def base64(self):
        return base64.b64encode(str(self.file_data).encode("utf-8"))

    def from_base64(self, base64_data):
        self.file_data = str(base64.b64decode(base64_data), 'UTF-8')

    def show(self, iter_times=3):
        data_resolve(self.file_data, iter_times, iter_times)

    def pickle_write(self, data_in):
        os.makedirs(str(Path(self.file_path).parent), exist_ok=True)
        with open(self.file_path, 'wb') as f:
            # protocol=pickle.HIGHEST_PROTOCOL
            pickle.dump(data_in, f)

    def json_write(self, data_in):
        if isinstance(data_in, dict):
            for k, v in data_in.items():
                if isinstance(v, np.bool_):
                    data_in[k] = bool(v)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_in, f)

    def npy_write(self, data_in):
        assert type(data_in) is np.array
        np.save(self.file_path, data_in)

    def npz_write(self, data_in):
        assert type(data_in) is dict
        np.savez(self.file_path, data_in)

    def pd_write(self, data_in, index=True):
        os.makedirs(str(Path(self.file_path).parent), exist_ok=True)
        data_in.to_csv(self.file_path, index=index)

    def yaml_dump(self, data_in):
        assert type(data_in) is dict
        import yaml
        with open(self.file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_in, f)
