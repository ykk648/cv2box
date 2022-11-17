# -- coding: utf-8 --
# @Time : 2022/10/26
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
"""
ref https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
"""
import concurrent.futures
from tqdm import tqdm


def tqdm_parallel_map(fn, *iterables, max_workers):
    """ use tqdm to show progress"""
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list)):
        yield f.result()


def thread_pool_wrapper(single_job_fn, data_list, max_workers=None):
    """
    multi cpu dispatcher
    """
    output = []
    for result in tqdm_parallel_map(single_job_fn, data_list, max_workers=max_workers):
        if result is not None:
            output += result
    return output
