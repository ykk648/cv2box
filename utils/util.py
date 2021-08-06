import os
import uuid


def oscall(command):
    print(command)
    os.system(command)


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]
