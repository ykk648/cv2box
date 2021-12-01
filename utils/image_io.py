# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

"""
skimage and pillow read image based uint8 and RGB mode
opencv read image based uint8 and BGR mode
using opencv as the default image read method
"""


class CVImage:
    def __init__(self, image_in, image_format=None):
        self.transform = None

        if not image_format:
            assert type(image_in) is str, 'if not give str path, name \'image_format\' !'
            self.cv_image = cv2.imread(image_in)
        elif 'cv' in image_format:
            self.cv_image = image_in
        elif 'pi' in image_format:
            self.cv_image = cv2.cvtColor(np.asarray(image_in), cv2.COLOR_RGB2BGR)
        elif 'sk' in image_format:
            self.cv_image = cv2.cvtColor((image_in * 255).astype(int), cv2.COLOR_RGB2BGR)
        elif 'ten' in image_format:
            image_numpy = image_in[0].cpu().float().numpy()
            self.cv_image = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            raise 'Can not find image_format ！'

    @property
    def rgb(self):
        return cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

    @property
    def bgr(self):
        return self.cv_image

    @property
    def pillow(self):
        return Image.fromarray(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))

    @property
    def tensor(self):
        import torch
        assert self.transform is not None, 'Use set_transform first !'
        img = self.transform(self.cv_image)
        return torch.unsqueeze(img, 0)

    def resize(self, size):
        if type(size) == tuple:
            self.cv_image = cv2.resize(self.cv_image, size)
        elif type(size) == int:
            self.cv_image = cv2.resize(self.cv_image, (size, size))
        else:
            raise 'Check the size input !'

    def set_transform(self, transform=None):
        if not transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def show(self):
        cv2.imshow('test', self.cv_image)
        cv2.waitKey(0)

    def save(self, img_save_p):
        cv2.imwrite(img_save_p, self.cv_image)


if __name__ == '__main__':
    pass
