import os
import uuid
import pickle


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
    f.close()
    return dummy

