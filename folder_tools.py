import os
import cv2
import shutil
from tqdm import tqdm


class FolderTools:
    def __init__(self, root_path, save_path):
        self.root_path = root_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def rename_folder_files(self, prefix1, prefix2):
        for f in tqdm(os.listdir(self.root_path)):
            f_p = os.path.join(self.root_path, f)
            aim_p = os.path.join(self.root_path, f.replace(prefix1, prefix2))
            os.rename(f_p, aim_p)

    def extract_imgs_from_folder(self, prefix1):
        for r, d, f in os.walk(self.root_path):
            for ff in f:
                if ff.split('.')[-1] in prefix1:
                    f_p = os.path.join(r, ff)
                    img = cv2.imread(f_p)
                    if img is not None:
                        shutil.copy(f_p, os.path.join(self.save_path, ff))

    def clean_one_folder_from_another(self, ):
        # clean root_p from save_p
        f_list = os.listdir(self.root_path)
        f_list_to_do = os.listdir(self.save_path)

        f_list_remove = list(set(f_list).intersection(set(f_list_to_do)))
        # print(len(f_list_remove))
        for f in tqdm(f_list_remove):
            os.remove(os.path.join(self.root_path, f))


if __name__ == '__main__':
    root_p = '/workspace/84_jiqun/home/user/luanjintai/projects/DeepFaceLab_Linux/素材/迪丽热巴/迪丽热巴_2'
    save_p = '/workspace/data/waifu_anime/idx1_out_2'
    folder_tools = FolderTools(root_p, save_p)
    folder_tools.rename_folder_files('.jpg', '_reba_2.jpg')
    # folder_tools.extract_imgs_from_folder(['jpg', 'png'])
    # folder_tools.clean_one_folder_from_another()
