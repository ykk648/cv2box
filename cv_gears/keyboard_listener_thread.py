# -- coding: utf-8 --
# @Time : 2021/11/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

from ..utils import try_import
import os

if os.environ['CV_MULTI_MODE'] == 'multi-thread':
    from multiprocessing.dummy import Process, Queue, Lock
elif os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock

try_import('pynput.keyboard', 'keyboard_listener: pip install pynput')


class KeyboardListener(Process):
    def __init__(self, share_list):
        super().__init__()
        self.share_list = share_list
        # do your init

    def run(self):
        # Collect events until released
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))

        except AttributeError:
            print('special key {0} pressed'.format(key))

    @staticmethod
    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False


if __name__ == '__main__':
    manager = Manager()
    share_list_ = manager.list()
    kl = KeyboardListener(share_list_)
