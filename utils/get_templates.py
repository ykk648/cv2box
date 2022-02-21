from utils import os_call, get_my_dir
import shutil
import os

tem_dict = {
    1: '{}/code_templates/multiprocess_consumer.py'.format(get_my_dir()),
    2: '{}/code_templates/multiprocess_by_concurrent.py'.format(get_my_dir()),
}


class GetTemplate:
    def __init__(self, t_serial=None):
        self.t = t_serial
        if self.t is None:
            print(tem_dict)
        else:
            self.copy_template()

    @staticmethod
    def show():
        help_msg = '1: multiprocess_consumer.py \n' \
                   '2: multiprocess_by_concurrent.py \n'
        print(help_msg)

    def copy_template(self):
        shutil.copyfile(tem_dict[self.t], os.getcwd())

