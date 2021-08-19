from utils import os_call
import shutil
import os

tem_dict = {
    1: './code_templates/multiprocess_consumer.py',
    2: './code_templates/multiprocess_consumer.py',
}


class GetTemplate:
    def __init__(self, t_serial=None):
        self.t = t_serial
        if self.t is None:
            print(tem_dict)
        else:
            self.copy_template()

    def copy_template(self):
        shutil.copyfile(tem_dict[self.t], os.getcwd())
        # os_call('cp ./shell_scripts/{}.sh \'{}\''.format(shell_name, dir_path))
