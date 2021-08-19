import os
import shutil
from tqdm import tqdm
from utils import os_call


def get_shell(dir_path, shell_name):
    os_call('cp ./shell_scripts/{}.sh \'{}\''.format(shell_name, dir_path))
    # shutil.copyfile('./shell_scripts/uncompress.sh', file_path + '/uncompress.sh')


def destroy_shell(dir_path, shell_name):
    os.remove('{}/{}.sh'.format(dir_path, shell_name))


def batch_uncompress(dir_path, pwd, pwd_flag=None):

    for f in os.listdir(dir_path):
        f_p = os.path.join(dir_path, f)
        if os.path.isdir(f_p):
            if f_p.find(pwd_flag) > 0:
                get_shell(f_p, 'uncompress')
                os_call('cd {} && sh uncompress.sh 7z ./ 202166'.format(f_p))
                destroy_shell(f_p, 'uncompress')
            else:
                get_shell(f_p, 'uncompress')
                os_call('cd {} && sh uncompress.sh tar ./'.format(f_p))
                destroy_shell(f_p, 'uncompress')


if __name__ == '__main__':
    p = ''
    # get_shell(p, 'uncompress')
    # destroy_shell(p, 'uncompress')
    batch_uncompress(p, '',pwd_flag='密码')