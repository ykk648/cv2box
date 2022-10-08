import os
import uuid
import pickle
import shutil
import time
from pathlib import Path
from importlib import import_module
import warnings
import sys
import numpy as np
from .logging import cv_print
import platform


def system_judge():
    """
    :return: e.g. windows linux java
    """
    return platform.system().lower()


def safe_cv_pyqt5():
    """
    resolve conflict made by opencv-python & pyqt5
    :return:
    """
    # ci_build_and_not_headless = False
    try:
        from cv2.version import ci_build, headless
        ci_and_not_headless = ci_build and not headless
    except:
        pass
    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_FONTDIR")


def os_call(command, silent=False):
    if silent:
        os.system(command + ' >/dev/null 2>&1')
    else:
        print(command)
        os.system(command)


def make_random_name(suffix_or_name=None):
    if '.' in suffix_or_name:
        return uuid.uuid4().hex + '.' + suffix_or_name.split('.')[-1]
    else:
        return uuid.uuid4().hex + '.' + suffix_or_name


def flush_print(str_to_print):
    print("\r" + "{}".format(str_to_print), end="", flush=True)


def get_my_dir():
    dir_name, filename = os.path.split(os.path.abspath(__file__))
    return dir_name


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


def mfc(flag='Your Func Name'):
    """
    A decorator to achieve MyTimer function
    :param flag:
    :return:
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            # print(f.__code__)
            res = f(*args, **kwargs)
            print('[{} {} fps: {fps}]'.format(flag, f.__name__, fps=1 / (time.time() - t0)))
            return res

        return wrapper

    return decorator


def get_path_by_ext(this_dir, ext_list=None, sorted_by_stem=False):
    if ext_list is None:
        print('Use image ext as default !')
        ext_list = [".jpg", ".png", ".JPG", ".webp", ".jpeg"]
    if sorted_by_stem:
        return sorted([p for p in Path(this_dir).rglob('*') if p.suffix in ext_list], key=lambda x: int(x.stem))
    else:
        return [p for p in Path(this_dir).rglob('*') if p.suffix in ext_list]


def try_import(module_name, warn_message=None):
    try:
        return import_module(module_name)
    except Exception as e:
        cv_print('got exception: {}, {}'.format(e, warn_message), 'error')
