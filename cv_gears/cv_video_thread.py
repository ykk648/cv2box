# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
from pathlib import Path
import os
import time
from tqdm import tqdm
import queue

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
        assert Path(video_in_path).exists()
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

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        with CVVideoLoader(self.video_path) as cvvl:
            for _ in tqdm(range(len(cvvl))):
                _, frame = cvvl.get()

                something_out = frame

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
