# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
"""
When using multi-process/torch-process, the network must be pickable and set multiprocess being 'spawn':
# import multiprocessing
# multiprocessing.set_start_method('spawn')
"""
import os
import time
import queue

from ..utils import cv_print as print

if os.environ['CV_MULTI_MODE'] == 'multi-thread':
    from multiprocessing.dummy import Process, Queue, Lock
elif os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock


class Factory(Process):
    def __init__(self, queue_list: list, fps_counter=False, block=True):
        super().__init__()
        assert len(queue_list) == 1
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        self.exit_signal = False

        # add init here
        print('Init {} {}, pid is {}.'.format('Factory', self.class_name(), self.pid_number))

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self):

        # do your work here.
        something_out = 0
        return something_out

    def exit_func(self):
        """
        Do your factory exit condition here.
        """
        self.exit_signal = False

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:

            # exit condition
            self.exit_func()
            if self.exit_signal:
                print('{} {} exit !'.format(self.class_name(), self.pid_number))
                self.queue_list[0].put(None)
                break

            try:
                something_out = self.forward_func()
            except Exception as e:
                print('{} raise error: {}'.format(self.class_name(), e))
                raise e

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if counter > 300:
                    print("{} FPS: {}".format(self.class_name(), counter / time_sum))
                    counter = 0
                    time_sum = 0

            if self.block:
                self.queue_list[0].put(something_out)
            else:
                try:
                    self.queue_list[0].put_nowait(something_out)
                except queue.Full:
                    # do your judge here, for example
                    queue_full_counter += 1
                    if (time.time() - start_time) > 10:
                        print('{} {} Queue full {} times'.format('Factory', self.class_name(), queue_full_counter))
            if self.fps_counter:
                start_time = time.time()


class Linker(Process):
    def __init__(self, queue_list: list, fps_counter=False, block=True):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        self.exit_signal = False

        # add init here
        print('init {} {}, pid is {}.'.format('Linker', self.class_name(), self.pid_number))

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self, something_in):

        # do your work here.
        something_out = something_in
        return something_out

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        print('{} {} exit !'.format(self.class_name(), self.pid_number))
        self.queue_list[1].put(None)
        self.exit_signal = True

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = 0

        while True:
            something_in = self.queue_list[0].get()
            if self.fps_counter:
                start_time = time.time()

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                break

            try:
                something_out = self.forward_func(something_in)
            except Exception as e:
                print('{} raise error:: {}'.format(self.class_name(), e))
                raise e

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if counter > 300:
                    print("{} FPS: {}".format(self.class_name(), counter / time_sum))
                    counter = 0
                    time_sum = 0

            if len(self.queue_list) > 1:
                if self.block:
                    self.queue_list[1].put(something_out)
                else:
                    try:
                        self.queue_list[1].put_nowait(something_out)
                    except queue.Full:
                        # do your judge here, for example
                        queue_full_counter += 1
                        if (time.time() - start_time) > 10:
                            print('{} {} Queue full {} times'.format('Linker', self.class_name(),
                                                                     queue_full_counter))


class Consumer(Process):
    def __init__(self, queue_list: list, fps_counter=False, block=True):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        self.exit_signal = False

        # add init here
        print('init {} {}, pid is {}.'.format('Consumer', self.class_name(), self.pid_number))

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self, something_in):
        # do your work here.
        something_out = something_in
        return something_out

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        print('{} {} exit !'.format(self.class_name(), self.pid_number))
        self.exit_signal = True

    def run(self, ):

        counter = 0
        time_sum = 0
        start_time = 0

        while True:
            something_in = self.queue_list[0].get()
            if self.fps_counter:
                start_time = time.time()

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                break

            try:
                self.forward_func(something_in)
            except Exception as e:
                print('{} raise error: {}'.format(self.class_name(), e))
                raise e

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if counter > 300:
                    print("{} FPS: {}".format(self.class_name(), counter / time_sum))
                    counter = 0
                    time_sum = 0
