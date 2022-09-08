import os
import logging
from ..utils.logging import cv_print as print

try:
    _ = os.environ['CV_MULTI_MODE']
except KeyError:
    os.environ['CV_MULTI_MODE'] = 'multi-thread'
    print('Use default multi mode: multi-thread, or you can set env \'CV_MULTI_MODE\' to '
           'multi-process/torch-process')

from .cv_video_thread import CVVideoThread
from .cv_multi_video_thread import CVMultiVideoThread
from .cv_threads_base import Factory, Linker, Consumer, Queue
# from .keyboard_listener_thread import KeyboardListener
