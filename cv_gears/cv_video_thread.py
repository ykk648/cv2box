# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
from pathlib import Path
import os
import time
from tqdm import tqdm
import queue
import cv2

if os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock
else:
    from multiprocessing.dummy import Process, Queue, Lock  # multi-thread

from ..cv_ops.cv_video import CVVideoLoader
from ..utils import cv_print as print


class CVVideoThread(Process):

    def __init__(self, video_in_path, queue_list: list, block=True, fps_counter=False):
        super().__init__()
        assert isinstance(video_in_path, int) or Path(video_in_path).exists()
        assert len(queue_list) == 1
        self.video_path = video_in_path
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        print('Init %s %s, pid is %s.', self.class_name(), self.__class__.__name__, self.pid_number)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def run(self, ):
        """
        Returns: BGR [frame]
        """
        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        with CVVideoLoader(self.video_path) as cvvl:
            # ref https://stackoverflow.com/questions/31472155/python-opencv-cv2-cv-cv-cap-prop-frame-count-get-wrong-numbers
            # cv2.CAP_PROP_FRAME_COUNT returns false count in some videos
            for _ in tqdm(range(len(cvvl) + 15)):
                success, frame = cvvl.get()
                if not success:
                    break

                something_out = [frame]

                if self.fps_counter:
                    counter += 1
                    time_sum += (time.time() - start_time)
                    if time_sum > 10:
                        print("%s FPS: %s", self.class_name(), counter / time_sum)
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
                            print('%s Queue full %s times', self.class_name(), queue_full_counter)
        self.queue_list[0].put(None)
        print('Video load done, %s exit', self.class_name())


class CVCamThread(Process):

    def __init__(self, video_in_path, queue_list: list, block=True, fps_counter=False):
        super().__init__()
        assert isinstance(video_in_path, int) or Path(video_in_path).exists()
        assert len(queue_list) == 1
        self.video_path = video_in_path
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        self.cap = cv2.VideoCapture(video_in_path)
        # self.cap.set(3, 1920)
        # self.cap.set(4, 1080)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # cap.set(cv2.CAP_PROP_FPS, 30)
        print('Init %s %s, pid is %s.', self.class_name(), self.__class__.__name__, self.pid_number)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def run(self, ):

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            _, frame = self.cap.read()

            something_out = [frame]

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("%s FPS: %s", self.class_name(), counter / time_sum)
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
                        print('%s Queue full %s times', self.class_name(), queue_full_counter)



class CVVideoWriterThread(Process):

    def __init__(self, video_writer, queue_list: list, block=True, fps_counter=False):
        super().__init__()
        assert len(queue_list) == 1
        self.video_writer = video_writer
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.pid_number = os.getpid()
        print('Init %s %s, pid is %s.', self.class_name(), self.__class__.__name__, self.pid_number)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def run(self, ):

        counter = 0
        time_sum = 0
        start_time = time.time()

        while True:
            something = self.queue_list[0].get()

            # exit condition
            if something is None:
                break

            src_img_in = something[0]
            self.video_writer.write(src_img_in)

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                if time_sum > 10:
                    print("%s FPS: %s", self.class_name(), counter / time_sum)
                    counter = 0
                    time_sum = 0
                start_time = time.time()

        print('Video save done, %s exit', self.class_name())
