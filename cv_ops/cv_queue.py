# -- coding: utf-8 --
# @Time : 2021/12/7
# @LastEdit : 2025/6/9
# @Author : ykk648

import time
import uuid
import numpy as np
from multiprocessing import shared_memory

# Global registry to track all shared memory objects
_shared_memory_registry = set()


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

        # Register the memory names in the global registry
        global _shared_memory_registry
        _shared_memory_registry.add(self.index_mem_name)
        _shared_memory_registry.add(self.data_size_name)

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

            # Register each data shared memory object
            global _shared_memory_registry
            _shared_memory_registry.add(data_shm_name)

        self.index_shm = shared_memory.ShareableList(self.data_shm_name_list, name=self.index_mem_name)
        self.data_size_shm = shared_memory.ShareableList(self.data_size_list, name=self.data_size_name)
        for i in range(self.queue_length):
            self.index_shm[i] = 'None'

    def push(self, push_buffer, aim_format=None):
        assert isinstance(push_buffer, bytes), 'push_buffer must be bytes'

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

    def get(self, timeout=None):
        """
        Get data from the shared memory queue

        Args:
            timeout: Maximum time to wait for data in seconds. None means wait indefinitely.

        Returns:
            get_buffer: SharedMemory object containing the data
            get_buffer_len: Length of the data in bytes

        Raises:
            TimeoutError: If timeout is reached and no data is available
        """
        start_time = time.time()
        while True:
            try:
                if self.index_shm[self.get_flag] != 'None':
                    # print(self.index_shm[self.get_flag])
                    get_buffer = shared_memory.SharedMemory(name=self.index_shm[self.get_flag])
                    # time.sleep(0.02)
                    get_buffer_len = self.data_size_shm[self.get_flag]
                    break

                # Check if timeout has been reached
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise TimeoutError("Timeout waiting for data in shared memory queue")

                time.sleep(self.rw_sleep_time)
            except ValueError:
                print('occur one mem access false, wait {}s and retry'.format(self.rw_sleep_time))
                time.sleep(self.rw_sleep_time)

                # Check if timeout has been reached
                if timeout is not None and (time.time() - start_time) > timeout:
                    raise TimeoutError("Timeout waiting for data in shared memory queue")

                continue
        return get_buffer, get_buffer_len

    def get_ok(self):
        self.index_shm[self.get_flag] = 'None'
        self.get_flag += 1
        self.get_flag %= self.queue_length

    def close(self):
        """Close and unlink all shared memory objects associated with this queue and clean up all registered shared memory"""
        global _shared_memory_registry

        # Close and unlink index shared memory
        try:
            if hasattr(self, 'index_shm') and self.index_shm is not None:
                try:
                    self.index_shm.shm.close()
                    self.index_shm.shm.unlink()
                    _shared_memory_registry.discard(self.index_mem_name)
                except (FileNotFoundError, BufferError) as e:
                    pass
        except Exception:
            pass

        # Close and unlink data size shared memory
        try:
            if hasattr(self, 'data_size_shm') and self.data_size_shm is not None:
                try:
                    self.data_size_shm.shm.close()
                    self.data_size_shm.shm.unlink()
                    _shared_memory_registry.discard(self.data_size_name)
                except (FileNotFoundError, BufferError) as e:
                    pass
        except Exception:
            pass

        # Close and unlink all data shared memory objects
        for i in range(len(self.data_shm_list)):
            try:
                if self.data_shm_list[i] is not None:
                    try:
                        name = self.data_shm_name_list[i] if i < len(self.data_shm_name_list) else None
                        self.data_shm_list[i].close()
                        self.data_shm_list[i].unlink()
                        if name:
                            _shared_memory_registry.discard(name)
                    except (FileNotFoundError, BufferError) as e:
                        pass
            except Exception:
                pass

        # Clean up all remaining shared memory objects
        if _shared_memory_registry:
            print(f"Cleaning up {len(_shared_memory_registry)} remaining shared memory objects...")
            for name in list(_shared_memory_registry):
                try:
                    # Try to directly unlink the shared memory
                    try:
                        shm = shared_memory.SharedMemory(name=name)
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
                    _shared_memory_registry.discard(name)
                except Exception:
                    pass

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
