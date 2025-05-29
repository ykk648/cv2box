# -- coding: utf-8 --
# @Time : 2022/6/28
# @LastEdit : 2025/6/9
# @Author : ykk648

from pathlib import Path
import os
import time
from tqdm import tqdm
import queue
import cv2
import subprocess

if os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock, Event
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock, Event
else:
    from multiprocessing.dummy import Process, Queue, Lock, Event  # multi-thread

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
        self._stop_event = Event()
        print('Init %s %s, pid is %s.', self.class_name(), self.__class__.__name__, self.pid_number)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def stop(self):
        self._stop_event.set()

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


class CVVideoCacheThread(CVVideoThread):
    def __init__(self, video_in_path, queue_list, need_more_frame_num, aim_output_frame_num, block=True, fps_counter=False):
        """
        Args:
            video_in_path:
            queue_list:
            need_more_frame_num: more frames needs to be cached while decoding video
            aim_output_frame_num: final output frame counts
            block:
            fps_counter:
        """
        super().__init__(video_in_path, queue_list, block=block, fps_counter=fps_counter)
        self.aim_output_frame_num = aim_output_frame_num
        self.need_more_frame_num = need_more_frame_num
        self.frame_caches = []

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
            for i in tqdm(range(len(cvvl) + 15)):
                success, frame = cvvl.get()
                if not success:
                    break

                if self.need_more_frame_num > len(cvvl):
                    self.frame_caches.append(frame)
                elif (len(cvvl) - i) < self.need_more_frame_num:
                    self.frame_caches.append(frame)

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

        circle_num = 0
        self.frame_caches = self.frame_caches[::-1]
        for j in tqdm(range(self.aim_output_frame_num - i)):
            if j // len(self.frame_caches) != circle_num:
                circle_num += 1
                self.frame_caches = self.frame_caches[::-1]
            self.queue_list[0].put([self.frame_caches[j % len(self.frame_caches)]])

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

    def __init__(self, video_writer, queue_list: list, block=True, fps_counter=False, counter_time=300):
        super().__init__()
        assert len(queue_list) == 1
        self.video_writer = video_writer
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.counter_time = counter_time
        self.block = block
        self.pid_number = os.getpid()
        print('Init %s %s, pid is %s.', self.class_name(), self.__class__.__name__, self.pid_number)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def run(self, ):

        counter = 0
        time_sum = 0

        while True:
            something = self.queue_list[0].get()
            if self.fps_counter:
                start_time = time.time()
            # exit condition
            if something is None:
                break

            src_img_in = something[0]

            if src_img_in is None:
                break

            if isinstance(self.video_writer, subprocess.Popen):
                try:
                    self.video_writer.stdin.write(src_img_in.tobytes())
                    # 检查进程是否还在运行
                    if self.video_writer.poll() is not None:
                        error_output = self.video_writer.stderr.read() if self.video_writer.stderr else 'Unknown error'
                        print(f'FFmpeg process terminated unexpectedly with return code {self.video_writer.returncode}')
                        print(f'Error output: {error_output}')
                        break
                except (BrokenPipeError, IOError) as e:
                    print(f'FFmpeg pipe error: {e}')
                    break
            elif isinstance(self.video_writer, cv2.VideoWriter):
                self.video_writer.write(src_img_in)
            else:
                raise NotImplementedError

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                time_sum = max(time_sum, 0.001)
                if counter > self.counter_time:
                    print("{} FPS: {}".format(self.class_name(), counter / time_sum))
                    counter = 0
                    time_sum = 0

        print('Video save done, %s exit', self.class_name())
