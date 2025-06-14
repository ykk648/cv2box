# -- coding: utf-8 --
# @Time : 2022/6/28
# @LastEdit : 2025/6/12
# @Author : ykk648

"""
When using multi-process/torch-process, the network must be pickable and set multiprocess being 'spawn':
# import multiprocessing
# multiprocessing.set_start_method('spawn')
"""
import os
import time
import queue
from typing import List, Any
import threading

from ..utils import cv_print as print

if os.environ['CV_MULTI_MODE'] == 'multi-thread':
    from multiprocessing.dummy import Process, Queue, Lock, Manager, Event
elif os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock, Manager, Event
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock, Manager, Event


class Factory(Process):
    def __init__(self, queue_list: List[Queue], fps_counter: bool = False, counter_time: int = 300, block: bool = True):
        super().__init__()
        # assert len(queue_list) == 1
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.counter_time = counter_time
        self.block = block
        self.pid_number = None  # 将在run()中设置
        self.thread_id = None   # 将在run()中设置
        self.exit_signal = False
        self._stop_event = Event()

        # add init here
        print(f'Init Factory {self.class_name()} in main thread')

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self):
        """
        Do your work here.
        Returns:
            Any: The output of the processing
        """
        something_out = 0
        return something_out

    def stop(self):
        self._stop_event.set()

    def exit_func(self):
        """
        Do your factory exit condition here.
        """
        self.exit_signal = False

    def run(self):
        # 在新进程中获取正确的ID
        self.pid_number = os.getpid()
        self.thread_id = threading.get_ident()
        print(f'Starting {self.class_name()}, pid={self.pid_number}, thread_id={self.thread_id}')

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()
        last_queue_warning = 0  # Track last warning time

        while not self._stop_event.is_set():
            # exit condition
            self.exit_func()
            if self.exit_signal:
                print(f'{self.class_name()} get exit signal !')
                self.queue_list[0].put(None)
                break

            try:
                something_out = self.forward_func()
            except Exception as e:
                print(f'{self.class_name()} raised error: {e}')
                # Optionally add traceback here
                import traceback
                print(traceback.format_exc())
                raise

            current_time = time.time()

            if self.fps_counter:
                counter += 1
                time_sum += (current_time - start_time)
                if counter > self.counter_time:
                    fps = counter / max(time_sum, 0.001)
                    print(f"{self.class_name()} FPS: {fps:.2f}")
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
                    # Only log warning every 10 seconds
                    if current_time - last_queue_warning > 10:
                        print(f'Factory {self.class_name()} Queue full {queue_full_counter} times in last {current_time - start_time:.1f} seconds.')
                        last_queue_warning = current_time

            if self.fps_counter:
                start_time = current_time

        print(f'Stop run func, {self.class_name()} {self.pid_number} {self.thread_id} exit !')


class Linker(Process):
    def __init__(self, queue_list: List[Queue], fps_counter: bool = False, counter_time: int = 300, block: bool = True, timeout=30):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.counter_time = counter_time
        self.block = block
        self.pid_number = None  # 将在run()中设置
        self.thread_id = None   # 将在run()中设置
        self.exit_signal = False
        self._stop_event = Event()
        self.timeout = timeout
        self.last_active_time = time.time()
        assert len(queue_list) > 1, "Linker queue list length must > 1"

        print(f'Init Linker {self.class_name()} in main thread')

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self, something_in: Any):
        """
        Do your work here.
        Args:
            something_in: Input data to process
        Returns:
            Any: Processed output data
        """
        something_out = something_in
        return something_out

    def stop(self):
        self._stop_event.set()

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        # Send None to all output queues
        for q in self.queue_list[1:]:
            try:
                q.put(None, timeout=5)
            except queue.Full:
                pass
        self.exit_signal = True

    def run(self):
        # 在新进程中获取正确的ID
        self.pid_number = os.getpid()
        self.thread_id = threading.get_ident()
        print(f'Starting {self.class_name()}, pid={self.pid_number}, thread_id={self.thread_id}')

        counter = 0
        time_sum = 0
        queue_full_counter = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                something_in = self.queue_list[0].get(timeout=self.timeout)
                if self.fps_counter:
                    start_time = time.time()
            except queue.Empty:
                print(f'{self.class_name()} {self.pid_number} {self.thread_id} timeout after {self.timeout} seconds')
                self.exit_func()
                break

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                print(f'{self.class_name()} get exit signal !')
                break

            try:
                something_out = self.forward_func(something_in)
            except Exception as e:
                print('{} raise error:: {}'.format(self.class_name(), e))
                raise e

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                time_sum = max(time_sum, 0.001)
                if counter > self.counter_time:
                    print("{} FPS: {}".format(self.class_name(), counter / time_sum))
                    counter = 0
                    time_sum = 0

            for output_queue in self.queue_list[1:]:
                try:
                    if self.block:
                        output_queue.put(something_out, timeout=self.timeout)
                    else:
                        output_queue.put_nowait(something_out)
                except queue.Full:
                    queue_full_counter += 1
                    print(f'{self.class_name()} {self.pid_number} put timeout/full after {self.timeout} seconds')
                    self.exit_func()
                    break

        print(f'Stop run func, {self.class_name()} {self.pid_number} {self.thread_id} exit !')


class Consumer(Process):
    def __init__(self, queue_list: List[Queue], fps_counter: bool = False, counter_time: int = 300, block: bool = True):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.counter_time = counter_time
        self.block = block
        self.pid_number = None  # 将在run()中设置
        self.thread_id = None   # 将在run()中设置
        self.exit_signal = False
        self._stop_event = Event()

        # add init here
        print(f'Init Consumer {self.class_name()} in main thread')

    @classmethod
    def class_name(cls):
        return cls.__name__

    def forward_func(self, something_in: Any):
        """
        Do your work here.
        Args:
            something_in: Input data to process
        """
        something_out = something_in
        return something_out

    def stop(self):
        self._stop_event.set()

    def exit_func(self):
        """
        If something is None, enter exit func, set `pass` if you want deal with exit by yourself.
        """
        self.exit_signal = True

    def run(self):
        # 在新进程中获取正确的ID
        self.pid_number = os.getpid()
        self.thread_id = threading.get_ident()
        print(f'Starting {self.class_name()}, pid={self.pid_number}, thread_id={self.thread_id}')

        counter = 0
        time_sum = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            something_in = self.queue_list[0].get()
            if self.fps_counter:
                start_time = time.time()

            # exit condition
            if something_in is None:
                self.exit_func()
            if self.exit_signal:
                print(f'{self.class_name()} get exit signal !')
                break

            try:
                self.forward_func(something_in)
            except Exception as e:
                print('{} raise error: {}'.format(self.class_name(), e))
                raise e

            if self.fps_counter:
                counter += 1
                time_sum += (time.time() - start_time)
                time_sum = max(time_sum, 0.001)
                if counter > self.counter_time:
                    print(f"{self.class_name()} FPS: {counter / time_sum}")
                    counter = 0
                    time_sum = 0

        print(f'Stop run func, {self.class_name()} {self.pid_number} {self.thread_id} exit !')


class QueueMonitor(Process):
    def __init__(self, queue_list: List[Queue], monitor_interval: float = 1.0,
                 alert_threshold: int = 100, name: str = "QueueMonitor"):
        """
        Initialize a queue monitor that periodically checks queue lengths.

        Args:
            queue_list: List of queues to monitor
            monitor_interval: How often to check queue lengths (in seconds)
            alert_threshold: Print warning if queue length exceeds this value
            name: Name of this monitor instance for logging
        """
        super().__init__()
        self.queue_list = queue_list
        self.monitor_interval = monitor_interval
        self.alert_threshold = alert_threshold
        self.monitor_name = name
        self._stop_event = Event()
        self.pid_number = None  # 将在run()中设置
        print(f'Init {self.monitor_name} in main thread')

    def run(self):
        # 在新进程中获取正确的ID
        self.pid_number = os.getpid()
        print(f'Starting {self.monitor_name}, pid={self.pid_number}')

        """Periodically check and report queue lengths."""
        while not self._stop_event.is_set():
            queue_sizes = []
            for i, q in enumerate(self.queue_list):
                size = q.qsize()
                queue_sizes.append(size)
                if size > self.alert_threshold:
                    print(f"⚠️ {self.monitor_name}: Queue {i} size ({size}) exceeds threshold {self.alert_threshold}")

            print(f"{self.monitor_name} - Queue sizes: {queue_sizes}")
            time.sleep(self.monitor_interval)

        print(f"{self.monitor_name} stopped.")

    def stop(self):
        """Stop the monitor."""
        self._stop_event.set()
