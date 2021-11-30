import os
import uuid
import pickle
import shutil
import numpy as np
import time


def os_call(command):
    print(command)
    os.system(command)


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]


def flush_print(str_to_print):
    print("\r" + "{}".format(str_to_print), end="", flush=True)


def pickle_load(pickle_path):
    with open(pickle_path, 'rb') as f:
        dummy = pickle.load(f)
    return dummy


def get_my_dir():
    dir_name, filename = os.path.split(os.path.abspath(__file__))
    return dir_name


# def give_me_ai_power():
#     shutil.copytree('{}/../AI_power'.format(get_my_dir()), './AI_power')


def np_norm(x):
    return (x - np.average(x)) / np.std(x)


class MyTimer(object):
    """
    timer
    """

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[finished, spent time: {time:.2f}s]'.format(time=time.time() - self.t0))
