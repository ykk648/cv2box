# -- coding: utf-8 --
# @Time : 2021/12/7
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import time
from multiprocessing import shared_memory
import uuid
import numpy as np


class CVQueue:
    def __init__(self, queue_length, mem_name, max_data_size=None, retry=True, rw_sleep_time=0.01, silence=False):
        self.push_buffer_len = None
        self.index_mem_name = mem_name
        self.data_size_name = mem_name + 'data_size'
        self.max_data_size = max_data_size
        self.queue_length = queue_length
        self.rw_sleep_time = rw_sleep_time
        self.push_flag = 0
        self.get_flag = 0

        self.data_shm_list = []
        self.index_shm = None
        self.data_size_shm = None
        self.data_shm_name_list = []
        self.data_size_list = []

        if not max_data_size:
            if retry:
                while True:
                    try:
                        self.index_shm = shared_memory.ShareableList(name=self.index_mem_name)
                    except FileNotFoundError:
                        if not silence:
                            print('can not find index mem name: {}, retry after 5s'.format(self.index_mem_name))
                        time.sleep(5)
                        continue
                    break
                while True:
                    try:
                        self.data_size_shm = shared_memory.ShareableList(name=self.data_size_name)
                    except FileNotFoundError:
                        if not silence:
                            print('can not find data size mem name: {}, retry after 5s'.format(self.data_size_name))
                        time.sleep(5)
                        continue
                    break
            else:
                self.index_shm = shared_memory.ShareableList(name=self.index_mem_name)
                self.data_size_shm = shared_memory.ShareableList(name=self.data_size_name)
            if not silence:
                print('this is guest, index mem name: {}, data size mem name: {}'.format(self.index_mem_name,
                                                                                         self.data_size_name))
        else:
            self.init_data_shm()
            if not silence:
                print('this is host, index mem name: {}, data size mem name: {}'.format(self.index_mem_name,
                                                                                        self.data_size_name))

    def init_data_shm(self):
        for i in range(self.queue_length):
            data_shm_name = uuid.uuid4().hex
            self.data_shm_list.append(
                shared_memory.SharedMemory(name=data_shm_name, create=True, size=self.max_data_size))
            self.data_shm_name_list.append(data_shm_name)
            self.data_size_list.append(self.max_data_size)
        self.index_shm = shared_memory.ShareableList(self.data_shm_name_list, name=self.index_mem_name)
        self.data_size_shm = shared_memory.ShareableList(self.data_size_list, name=self.data_size_name)
        for i in range(self.queue_length):
            self.index_shm[i] = 'None'

    def put(self, push_buffer, aim_format=None):
        # if aim_format == 'numpy':
        #     push_buffer = push_buffer.tobytes()

        while True:
            self.push_buffer_len = len(push_buffer)
            if self.index_shm[self.push_flag] == 'None':
                self.data_shm_list[self.push_flag].buf[:self.push_buffer_len] = push_buffer[:]
                break
            time.sleep(self.rw_sleep_time)
        self.index_shm[self.push_flag] = self.data_shm_name_list[self.push_flag]
        self.data_size_shm[self.push_flag] = self.push_buffer_len
        self.push_flag += 1
        self.push_flag %= self.queue_length

    # def put_ok(self):
    #     pass

    def get(self):
        while True:
            try:
                if self.index_shm[self.get_flag] != 'None':
                    # print(self.index_shm[self.get_flag])
                    get_buffer = shared_memory.SharedMemory(name=self.index_shm[self.get_flag])
                    # time.sleep(0.02)
                    get_buffer_len = self.data_size_shm[self.get_flag]
                    break
                time.sleep(self.rw_sleep_time)
            except ValueError:
                print('occur one mem access false, wait {}s and retry'.format(self.rw_sleep_time))
                time.sleep(self.rw_sleep_time)
                continue
        return get_buffer, get_buffer_len

    def get_ok(self):
        self.index_shm[self.get_flag] = 'None'
        self.get_flag += 1
        self.get_flag %= self.queue_length

    def close(self):
        try:
            self.index_shm.shm.close()
            self.index_shm.shm.unlink()
        except FileNotFoundError:
            return
        try:
            self.data_size_shm.shm.close()
            self.data_size_shm.shm.unlink()
        except FileNotFoundError:
            return
        for i in range(len(self.data_shm_list)):
            try:
                self.data_shm_list[i].close()
                self.data_shm_list[i].unlink()
            except FileNotFoundError:
                return

    def full(self):
        time.sleep(self.rw_sleep_time)
        try:
            return not (np.array(self.index_shm) == 'None').any()
        except ValueError:
            print('occur one mem access false, wait {}s and retry'.format(self.rw_sleep_time))
            time.sleep(self.rw_sleep_time)
            return not (np.array(self.index_shm) == 'None').any()

    def empty(self):
        time.sleep(self.rw_sleep_time)
        try:
            return (np.array(self.index_shm) == 'None').all()
        except ValueError:
            print('occur one mem access false, wait {}s and retry'.format(self.rw_sleep_time))
            time.sleep(self.rw_sleep_time)
            return (np.array(self.index_shm) == 'None').all()

    @staticmethod
    def clean_mem(mem_list: list, silence=True):
        for name in mem_list:
            try:
                CVQueue(10, mem_name=name, retry=False, silence=True).close()
            except:
                pass
        if not silence:
            print('clean mem \'{} \'done !'.format(mem_list))

