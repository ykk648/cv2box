# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

from pathlib import Path
import os
import time
from tqdm import tqdm
import queue
import datetime
import numpy as np
from multiprocessing.dummy import Process, Queue, Lock

from .vidgear import CamGear
from ..cv_ops import CVImage


class ReconnectingCamGear:
    def __init__(self, source_list_, reset_attempts=50, reset_delay=5, multi_stream_offset=True):
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.source_list = source_list_
        self.multi_stream_offset = multi_stream_offset

        self.input_option_dict = {
            # 'CAP_PROP_FRAME_WIDTH': 1280,
            # 'CAP_PROP_FRAME_HEIGHT': 720,
            # 'CAP_PROP_FPS': 30,
            # # ('M', 'P', '4', '2') ('X', 'V', 'I', 'D') ('H', '2', '6', '4')
            # 'CAP_PROP_FOURCC': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            'THREADED_QUEUE_MODE': True,
        }

        # start_time = time.time()
        self.sourceA = CamGear(source=self.source_list[0], logging=True, time_delay=0, **self.input_option_dict).start()
        a_time = self.sourceA.get_first_grab_time()
        self.sourceB = CamGear(source=self.source_list[1], logging=True, time_delay=0, **self.input_option_dict).start()
        b_time = self.sourceB.get_first_grab_time()
        self.sourceC = CamGear(source=self.source_list[2], logging=True, time_delay=0, **self.input_option_dict).start()
        c_time = self.sourceC.get_first_grab_time()
        self.sourceD = CamGear(source=self.source_list[3], logging=True, time_delay=0, **self.input_option_dict).start()
        d_time = self.sourceD.get_first_grab_time()

        if self.multi_stream_offset:
            self.pass_frame_number_a = round((d_time - a_time) / (1 / 30))  # int round
            self.pass_frame_number_b = round((d_time - b_time) / (1 / 30))
            self.pass_frame_number_c = round((d_time - c_time) / (1 / 30))
            print('frame offset: ', self.pass_frame_number_a, self.pass_frame_number_b, self.pass_frame_number_c)

        self.running = True
        self.first = True
        # self.framerateA = self.sourceA.framerate

    def read(self):
        if self.sourceA is None or self.sourceB is None:
            return None
        if self.running and self.reset_attempts > 0:
            if self.first and self.multi_stream_offset:
                # offset
                while self.pass_frame_number_a > 0:
                    _ = self.sourceA.read()
                    self.pass_frame_number_a -= 1
                while self.pass_frame_number_b > 0:
                    _ = self.sourceB.read()
                    self.pass_frame_number_b -= 1
                while self.pass_frame_number_c > 0:
                    _ = self.sourceC.read()
                    self.pass_frame_number_c -= 1
                self.first = False

            frameA = self.sourceA.read()
            frameB = self.sourceB.read()
            frameC = self.sourceC.read()
            frameD = self.sourceD.read()

            if frameA is None or frameB is None or frameC is None or frameD is None:
                self.sourceA.stop()
                self.sourceB.stop()
                self.sourceC.stop()
                self.sourceD.stop()
                self.reset_attempts -= 1
                if self.reset_attempts == 0:
                    return None, None, None, None
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.sourceA = CamGear(source=self.source_list[0]).start()
                self.sourceB = CamGear(source=self.source_list[1]).start()
                self.sourceC = CamGear(source=self.source_list[2]).start()
                self.sourceD = CamGear(source=self.source_list[3]).start()
                # return previous frame
                return self.frameA, self.frameB, self.frameC, self.frameD
            else:
                self.frameA = frameA
                self.frameB = frameB
                self.frameC = frameC
                self.frameD = frameD
                return frameA, frameB, frameC, frameD
        else:
            return None, None, None, None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.sourceA is None and not self.sourceB is None and not self.sourceC is None and not self.sourceD is None:
            self.sourceA.stop()
            self.sourceB.stop()
            self.sourceC.stop()
            self.sourceD.stop()


class CVMultiVideoThread(Process):

    def __init__(self, video_in_path_list, queue_list: list, multi_stream_offset=True, silent=False, block=True,
                 fps=None, fps_counter=False, process_name='CVMultiVideoThread'):
        super().__init__()
        assert len(queue_list) == 1
        assert len(video_in_path_list) == 4
        self.video_in_path_list = video_in_path_list

        self.rcg = ReconnectingCamGear(video_in_path_list, reset_attempts=1, reset_delay=5,
                                       multi_stream_offset=multi_stream_offset)

        self.queue_list = queue_list
        self.silent = silent
        self.fps_counter = fps_counter
        self.block = block
        self.process_name = process_name
        self.pid_number = os.getpid()
        self.fps = fps
        if not self.silent:
            print('init {} {}, pid is {}.'.format(self.process_name, self.__class__.__name__, self.pid_number))

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            one_start_time = time.time()
            frameA_, frameB_, frameC_, frameD_ = self.rcg.read()

            if frameA_ is None or frameB_ is None or frameC_ is None or frameD_ is None:
                self.queue_list[0].put(None)
                self.rcg.stop()
                break

            something_out = [frameA_, frameB_, frameC_, frameD_]

            # CVImage(frameA_).show(1)

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("{} FPS: {}".format(self.process_name, counter / time_sum))
                    counter = 0
                    time_sum = 0
                start_time = time.time()

            if self.block:
                # for i in range(len(something_out)):
                self.queue_list[0].put(something_out)
            else:
                try:
                    self.queue_list[0].put_nowait(something_out)
                except queue.Full:
                    # do your judge here, for example
                    queue_full_counter += 1
                    if (time.time() - start_time) > 10:
                        print('{} Queue full {} times'.format(self.process_name, queue_full_counter))
            if self.fps is not None and self.fps > 0:
                time.sleep(np.max([1 / self.fps - (time.time() - one_start_time), 0]))
        # self.queue_list[0].put(None)
        # if not self.silent:
        #     print('Video load done, {} exit'.format(self.process_name))


if __name__ == '__main__':
    source_list = [0, 2, 4, 6]
    q1 = Queue(5)

    cvmt = CVMultiVideoThread(source_list, [q1])
    cvmt.start()

    while True:
        frameA_, frameB_, frameC_, frameD_ = q1.get()
        CVImage(frameB_).show(1)
