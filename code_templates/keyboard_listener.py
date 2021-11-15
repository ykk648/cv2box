# -- coding: utf-8 --
# @Time : 2021/11/15
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from pynput import keyboard
from multiprocessing.dummy import Process, Queue, Array, Manager


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
