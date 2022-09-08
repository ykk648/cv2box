# -- coding: utf-8 --
# @Time : 2022/8/18
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

import logging
import os
from tqdm.contrib.logging import logging_redirect_tqdm

CV_LOG_LEVEL = os.environ['CV_LOG_LEVEL']

LEVEL_DICT = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 30,
    'critical': 50,
}


def cv_print(message, *args, level='info'):
    logger = logging.getLogger('cv2box')
    with logging_redirect_tqdm(loggers=[logger]):
        if level == 'debug':
            logger.debug(message, *args)
        elif level == 'info':
            logger.info(message, *args)
        elif level == 'warning':
            logger.warning(message, *args)
        elif level == 'error':
            logger.error(message, *args)
        elif level == 'critical':
            logger.critical(message, *args)


def set_log_level(level='info'):
    logger = logging.getLogger('cv2box')
    logger.setLevel(LEVEL_DICT[level])


def judge_log_level(level='info'):
    logger = logging.getLogger('cv2box')
    level_now = logger.getEffectiveLevel()
    return level_now == LEVEL_DICT[level]


def cv_logging_init():
    logger = logging.getLogger('cv2box')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()

    ch.setLevel(LEVEL_DICT[CV_LOG_LEVEL])

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
