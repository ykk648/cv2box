# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

# from multiprocessing import Process, Lock, Queue # multi process
from multiprocessing.dummy import Process, Queue  # multi thread
import os
import time
import queue


class Factory(Process):
    def __init__(self, queue_list: list, block=True, fps_counter=False, process_name='Factory'):
        super().__init__()
        assert len(queue_list) == 1
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.process_name = process_name
        self.pid = os.getpid()

        # add init here

        print('init {} {}, pid is {}.'.format(self.process_name, self.__class__.__name__, self.pid))

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
                print('{} {} exit !'.format(self.__class__.__name__, self.pid))
                self.queue_list[0].put(None)
                break

            try:
                something_out = self.forward_func()
            except Exception as e:
                print('{} raise error {}'.format(self.process_name, e))

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("{} FPS: {}".format(self.process_name, counter / time_sum))
                    counter = 0
                    time_sum = 0
                start_time = time.time()

            if self.block:
                self.queue_list[0].put(something_out)
            else:
                try:
                    self.queue_list[0].put_nowait(something_out)
                except queue.Full:
                    # do your judge here, for example
                    queue_full_counter += 1
                    if (time.time() - start_time) > 10:
                        print('{} Queue full {} times'.format(self.process_name, queue_full_counter))


class Linker(Process):
    def __init__(self, queue_list: list, block=True, fps_counter=False, process_name='Linker'):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.process_name = process_name
        self.pid = os.getpid()
        self.exit_signal = False

        # add init here

        print('init {} {}, pid is {}.'.format(self.process_name, self.__class__.__name__, self.pid))

    def forward_func(self, something_in):

        # do your work here.
        something_out = something_in
        return something_out

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        print('{} {} exit !'.format(self.__class__.__name__, self.pid))
        self.queue_list[1].put(None)
        self.exit_signal = True

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                break

            try:
                something_out = self.forward_func(something_in)
            except Exception as e:
                print('{} raise error {}'.format(self.process_name, e))

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("{} FPS: {}".format(self.process_name, counter / time_sum))
                    counter = 0
                    time_sum = 0
                start_time = time.time()

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
                            print('{} {} Queue full {} times'.format(self.process_name, self.__class__.__name__,
                                                                     queue_full_counter))


class Consumer(Process):
    def __init__(self, queue_list: list, block=True, fps_counter=False, process_name='Factory'):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.process_name = process_name
        self.pid = os.getpid()
        self.exit_signal = False

        # add init here

        print('init {} {}, pid is {}.'.format(self.process_name, self.__class__.__name__, self.pid))

    def forward_func(self, something_in):

        # do your work here.
        something_out = something_in
        return something_out

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        print('{} {} exit !'.format(self.__class__.__name__, self.pid))
        self.exit_signal = True

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                break

            try:
                self.forward_func(something_in)
            except Exception as e:
                print('{} raise error {}'.format(self.process_name, e))

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("{} FPS: {}".format(self.process_name, counter / time_sum))
                    counter = 0
                    time_sum = 0
                start_time = time.time()


if __name__ == '__main__':
    q1 = Queue(10)
    q2 = Queue(10)
    c1 = Consumer([q1, q2])
    c1.start()
