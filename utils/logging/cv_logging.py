# -- coding: utf-8 --
# @Time : 2022/8/18
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

import logging
import os

CV_LOG_LEVEL = os.environ['CV_LOG_LEVEL']

LEVEL_DICT = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 30,
    'critical': 50,
}


def cv_log(message, level='info'):
    logger = logging.getLogger('cv2box')
    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)


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
