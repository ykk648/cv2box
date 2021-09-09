import os
import cv2
import shutil
from tqdm import tqdm
import random
from .utils import make_random_name
from pathlib import Path


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

    def extract_imgs_from_folder(self, img_num=0, patten=None, random_flag=True):
        if patten is None:
            patten = '*.jpg'
        count = 0
        path = Path(self.root_path)
        file_list = list(path.rglob(patten))
        if random_flag:
            random.shuffle(file_list)
        for f_p in file_list:
            # print(f_p)
            img = cv2.imread(str(f_p))
            if img is not None:
                ff = f_p.name
                aim_p = os.path.join(self.save_path, ff)
                if os.path.exists(aim_p):
                    aim_p = os.path.join(self.save_path, make_random_name(ff))
                shutil.copy(f_p, aim_p)
                count += 1
                if count % 1000 == 0:
                    print('copy {} imgs from {} to {}'.format(count, self.root_path, self.save_path))
                if count == img_num:
                    return

    def clean_one_folder_from_another(self, ):
        # clean root_p from save_p
        f_list = os.listdir(self.root_path)
        f_list_to_do = os.listdir(self.save_path)

        f_list_remove = list(set(f_list).intersection(set(f_list_to_do)))
        # print(len(f_list_remove))
        for f in tqdm(f_list_remove):
            os.remove(os.path.join(self.root_path, f))
