# -- coding: utf-8 --
# @Time : 2021/12/7
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import time
from multiprocessing import shared_memory
import uuid
import numpy as np
from cv2box import CVImage
import array


class CVQueue:
    def __init__(self, queue_length, mem_name, max_data_size=None):
        self.index_mem_name = mem_name
        self.data_size_name = mem_name + 'data_size'
        self.max_data_size = max_data_size
        self.queue_length = queue_length
        self.push_flag = 0
        self.get_flag = 0

        self.data_shm_list = []
        self.index_shm = None
        self.data_size_shm = None
        self.data_shm_name_list = []
        self.data_size_list = []

        try:
            self.index_shm = shared_memory.ShareableList(name=self.index_mem_name)
            self.data_size_shm = shared_memory.ShareableList(name=self.data_size_name)
            print('this is guest')
        except FileNotFoundError:
            self.init_data_shm()
            print('this is host')

    def init_data_shm(self):
        for i in range(self.queue_length):
            data_shm_name = uuid.uuid4().hex
            self.data_shm_list.append(shared_memory.SharedMemory(name=data_shm_name, create=True, size=self.max_data_size))
            self.data_shm_name_list.append(data_shm_name)
            self.data_size_list.append(self.max_data_size)
        self.index_shm = shared_memory.ShareableList(self.data_shm_name_list, name=self.index_mem_name)
        self.data_size_shm = shared_memory.ShareableList(self.data_size_list, name=self.data_size_name)
        for i in range(self.queue_length):
            self.index_shm[i] = 'None'

    def put(self, push_buffer):
        while True:
            push_buffer_len = len(push_buffer)
            if self.index_shm[self.push_flag] == 'None':
                self.data_shm_list[self.push_flag].buf[:push_buffer_len] = push_buffer[:]
                self.index_shm[self.push_flag] = self.data_shm_name_list[self.push_flag]
                self.data_size_shm[self.push_flag] = push_buffer_len
                break
        self.push_flag += 1
        self.push_flag %= self.queue_length

    def get(self):
        while True:
            if self.index_shm[self.get_flag] != 'None':
                print(self.index_shm[self.get_flag])
                get_buffer = shared_memory.SharedMemory(name=self.index_shm[self.get_flag])
                get_buffer_len = self.data_size_shm[self.get_flag]
                break
        return get_buffer, get_buffer_len

    def get_ok(self):
        self.index_shm[self.get_flag] = 'None'
        self.get_flag += 1
        self.get_flag %= self.queue_length

    def close(self):
        self.index_shm.shm.close()
        self.data_size_shm.shm.close()
        self.index_shm.shm.unlink()
        self.data_size_shm.shm.unlink()
        for i in range(len(self.data_shm_list)):
            self.data_shm_list[i].close()
            self.data_shm_list[i].unlink()

    def full(self):
        return not (np.array(self.index_shm) == 'None').any()

    def empty(self):
        return (np.array(self.index_shm) == 'None').all()

if __name__ == '__main__':
    # 50M
    cvq = CVQueue(10, mem_name='okbb', max_data_size=50*1024*1024)

    from utils.util import get_path_by_ext

    for img_p in get_path_by_ext(''):
        img_buffer = CVImage(img_p).format_bytes
        print(len(img_buffer))
        print(cvq.full())
        print(cvq.empty())
        cvq.push(img_buffer)

        # get_buf = cvq.get()
        # print(get_buf)
        # print(type(get_buf.buf))
        # CVImage(bytes(get_buf.buf), image_format='bytes').show()
