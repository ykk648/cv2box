# -- coding: utf-8 --
# @Time : 2021/11/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

from ..utils import try_import
import os

if os.environ['CV_MULTI_MODE'] == 'multi-thread':
    from multiprocessing.dummy import Process, Queue, Lock, Manager
elif os.environ['CV_MULTI_MODE'] == 'multi-process':
    from multiprocessing import Process, Queue, Lock, Manager
elif os.environ['CV_MULTI_MODE'] == 'torch-process':
    from torch.multiprocessing import Process, Queue, Lock, Manager

keyboard = try_import('pynput.keyboard', 'keyboard_listener: pip install pynput')


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


class KeyboardListenerStr(KeyboardListener):
    def __init__(self, share_list):
        super().__init__(share_list)
        # do your init
        self.current_keys = []

    def run(self):
        # Collect events until released
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

    @staticmethod
    def format_key(key):
        # 根据实际需求格式化按键，你可以根据需要进行修改
        if isinstance(key, keyboard.KeyCode):
            return key.char
        else:
            return str(key)

    def on_press(self, key):
        if key != keyboard.Key.esc:
            self.current_keys.append(self.format_key(key))

    def on_release(self, key):
        if key == keyboard.Key.enter:
            print('Current input: ', ''.join(self.current_keys))
            self.current_keys.clear()


if __name__ == '__main__':
    manager = Manager()
    share_list_ = manager.list()
    kl = KeyboardListener(share_list_)
