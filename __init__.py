# VERSION 0.5.8
import os

try:
    _ = os.environ['CV_LOG_LEVEL']
except KeyError:
    os.environ['CV_LOG_LEVEL'] = 'info'
    from .utils import cv_print
    # cv_print('Use default log level: info, or you can set env \'CV_LOG_LEVEL\' to '
    #        'debug/info/warning/error/critical', level='info')

from .utils import MyTimer, MyFpsCounter, mfc, try_import, get_path_by_ext, set_log_level, system_judge
from .cv_ops import CVImage, CVQueue, CVFile, CVVideo, CVVideoLoader, CVVideoMaker, CVExcel, CVFolder, CVBbox, CVCamera
