# from multiprocessing import Process, Lock, Queue # multi process
from multiprocessing.dummy import Process, Queue  # multi thread
import os
import time
import queue

try:
    from cv2box import flush_print as fp
except ModuleNotFoundError:
    fp = print


class Consumer(Process):
    def __init__(self, queue_list: list, block=True, fps_counter=False):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid = os.getpid()

        # add init here

        print('init consumer {}, pid is {}.'.format(self.__class__.__name__, self.pid))

    def forward_func(self, something_in):

        # do your work here.
        something_out = something_in
        return something_out

    def run(self,):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            # exit condition
            if something_in is None:
                print('subprocess {} exit !'.format(self.pid))
                break

            something_out = self.forward_func(something_in)

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if (time.time() - start_time) > 10:
                    fp("Consumer FPS: {}".format(counter / time_sum))
                    counter = 0
                start_time = time.time()

            if self.block:
                self.queue_list[1].put(something_out)
            else:
                try:
                    self.queue_list[1].put_nowait(something_out)
                except queue.Full:
                    # do your judge here, for example
                    queue_full_counter += 1
                    if (time.time() - start_time) > 10:
                        fp('Queue full {} times'.format(queue_full_counter))




if __name__ == '__main__':
    q1 = Queue(10)
    q2 = Queue(10)
    c1 = Consumer([q1, q2])
    c1.start()
