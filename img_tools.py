from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import cv2


# import imageio


class ImgTools:
    def __init__(self, root_path, save_path):
        # self.img_path = img_path
        self.root_path = root_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def alpha_channel_white(self, ):
        for f in tqdm(os.listdir(self.root_path)):
            f_p = os.path.join(self.root_path, f)
            pic = Image.open(f_p)
            pic = pic.convert('RGBA')  # 转为RGBA模式
            width, height = pic.size
            array = pic.load()  # 获取图片像素操作入口
            for i in range(width):
                for j in range(height):
                    pos = array[i, j]  # 获得某个像素点，格式为(R,G,B,A)元组
                    try:
                        isEdit = (pos[3] == 0)
                        if isEdit:
                            array[i, j] = (255, 255, 255, 0)
                    except IndexError or TypeError:
                        break
            pic.save(os.path.join(self.save_path, f), 'PNG')

    def convert_2_3channel(self, ):
        for f in tqdm(os.listdir(self.root_path)):
            f_p = os.path.join(self.root_path, f)
            img = Image.open(f_p)
            # 将一个4通道转化为rgb三通道
            img = img.convert("RGB")
            img = np.array(img, dtype=np.uint8)[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(self.save_path, f), img)

    def resize_folder_img(self, aim_size, channel):
        for f in tqdm(os.listdir(self.root_path)):
            f_aim_p = os.path.join(self.save_path, f)
            if not os.path.exists(f_aim_p):
                f_p = os.path.join(self.root_path, f)
                img = Image.open(f_p)
                # img = img.convert("RGB")
                img = np.array(img, dtype=np.uint8)
                if channel == 4:
                    img = cv2.resize(img, aim_size)[:, :, [2, 1, 0, 3]]
                else:
                    img = cv2.resize(img, aim_size)[:, :, [2, 1, 0]]

                cv2.imwrite(f_aim_p, img)
                # img.save(os.path.join(self.save_path, f))

    @staticmethod
    def cv_show(img_path):
        img = cv2.imread(img_path)
        cv2.imshow("original", img)
        cv2.waitKey(0)

    def gen_pure_color_img(self, aim_size, color_, save_path):

        img = np.ones((aim_size[0], aim_size[1]), dtype=np.uint8)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('bgr_img', bgr_img)
        bgr_img[:, :, 0] = color_[0]
        bgr_img[:, :, 1] = color_[1]
        bgr_img[:, :, 2] = color_[2]
        cv2.imwrite(save_path + '/' + str(color_) + '.png', bgr_img)
