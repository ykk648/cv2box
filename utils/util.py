import os
import uuid
import pickle
import shutil
import numpy as np
import time
from pathlib import Path
from importlib import import_module
import warnings

def os_call(command):
    print(command)
    os.system(command)


def make_random_name(suffix_or_name=None):
    if '.' in suffix_or_name:
        return uuid.uuid4().hex + '.' + suffix_or_name.split('.')[-1]
    else:
        return uuid.uuid4().hex + '.' + suffix_or_name


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


# def np_norm(v):
#     norm = np.linalg.norm(v)
#     if norm == 0:
#         return v
#     return v / norm

class MyTimer(object):
    """
    timer
    """

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[finished, spent time: {time:.2f}s]'.format(time=time.time() - self.t0))


class MyFpsCounter(object, ):
    def __init__(self, flag='temp'):
        self.flag = flag

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[{} fps: {fps}]'.format(self.flag, fps=1 / (time.time() - self.t0)))


def get_path_by_ext(this_dir, ext_list=None):
    if ext_list is None:
        print('Use image ext as default !')
        ext_list = [".jpg", ".png", ".JPG", ".webp", ".jpeg"]
    return [p for p in Path(this_dir).rglob('*') if p.suffix in ext_list]


def try_import(pkg_name):
    try:
        import_module(pkg_name)
    except ModuleNotFoundError:
        warnings.warn('can not find package: {}, try install or reinstall it !'.format(pkg_name), )
        pass
