import os

try:
    CV_MULTI_MODE = os.environ['CV_MULTI_MODE']
except KeyError:
    os.environ['CV_MULTI_MODE'] = 'multi-thread'
    print('cv2box: Use default multi mode: multi-thread, or you can set env \'CV_MULTI_MODE\' to '
          'multi-process/torch-process')

from .cv_video_thread import CVVideoThread
from .cv_multi_video_thread import CVMultiVideoThread
from .cv_threads_parts import Factory, Linker, Consumer
from .cv_threads_parts import Queue
