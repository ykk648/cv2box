import os
import cv2
import shutil
from tqdm import tqdm
import random
from .utils import make_random_name


class FolderTools:
    def __init__(self, root_path, save_path=None):
        self.root_path = root_path
        self.save_path = save_path
        if save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def rename_folder_files(self, prefix1, prefix2):
        for f in tqdm(os.listdir(self.root_path)):
            f_p = os.path.join(self.root_path, f)
            aim_p = os.path.join(self.root_path, f.replace(prefix1, prefix2))
            os.rename(f_p, aim_p)

    def extract_imgs_from_folder(self, prefix1=None, img_num=0, random_flag=True):
        if prefix1 is None:
            prefix1 = ['jpg', 'png']
        count = 0
        for r, d, f in os.walk(self.root_path):
            if random_flag:
                random.shuffle(f)
            for ff in f:
                if ff.split('.')[-1] in prefix1:
                    f_p = os.path.join(r, ff)
                    img = cv2.imread(f_p)
                    if img is not None:
                        aim_p = os.path.join(self.save_path, ff)
                        if os.path.exists(aim_p):
                            aim_p = os.path.join(self.save_path, make_random_name(ff))
                        shutil.copy(f_p, aim_p)
                        count += 1
                        if count == img_num:
                            print('copy {} imgs from {} to {}'.format(count, self.root_path, self.save_path))
                            return

    def clean_one_folder_from_another(self, ):
        # clean root_p from save_p
        f_list = os.listdir(self.root_path)
        f_list_to_do = os.listdir(self.save_path)

        f_list_remove = list(set(f_list).intersection(set(f_list_to_do)))
        # print(len(f_list_remove))
        for f in tqdm(f_list_remove):
            os.remove(os.path.join(self.root_path, f))

