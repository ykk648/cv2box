# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import cv2
import numpy as np
import base64
import io
from pathlib import PosixPath

"""
skimage and pillow read image based uint8 and RGB mode
opencv read image based uint8 and BGR mode
using opencv as the default image read method
"""


class ImageBasic:
    def __init__(self, image_in, image_format, image_size):
        if isinstance(image_in, PosixPath):
            image_in = str(image_in)
        if isinstance(image_in, str) and image_format == 'cv2':
            # assert type(image_in) is str, 'if not give str path, name \'image_format\' !'
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
        elif 'base' in image_format:
            img_data = base64.b64decode(image_in[22:])
            img_array = np.frombuffer(img_data, np.uint8)
            self.cv_image = cv2.imdecode(img_array, 1)
        elif 'byte' in image_format:
            self.cv_image = cv2.imdecode(np.frombuffer(io.BytesIO(image_in).read(), np.uint8), 1)
        elif 'buffer' in image_format:
            self.cv_image = np.frombuffer(image_in, np.uint8).reshape(image_size)
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
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))

    def resize(self, size):
        if type(size) == tuple:
            if size != self.cv_image.shape[:-1]:
                # cv2 resize function always returns a new Mat object.
                self.cv_image = cv2.resize(self.cv_image, size)
        elif type(size) == int:
            if size != self.cv_image.shape[0]:
                self.cv_image = cv2.resize(self.cv_image, (size, size))
        else:
            raise 'Check the size input !'
        return self

    def show(self, window_name='test'):
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.cv_image)
        cv2.waitKey(0)

    def save(self, img_save_p):
        cv2.imwrite(img_save_p, self.cv_image)


class CVImage(ImageBasic):
    def __init__(self, image_in, image_format='cv2', image_size=None):
        super().__init__(image_in, image_format, image_size)
        self.transform = None
        self.input_std = self.input_mean = self.input_size = None

    # ===== for image transfer =====
    @property
    def base64(self):
        """
        :return: jpg format base64 code
        """
        image = cv2.imencode('.jpg', self.cv_image)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        return 'data:image/jpg;base64,' + image_code

    @property
    def bytes(self):
        """
        fast enough for video real-time steam process
        :return:
        """
        return self.cv_image.tobytes()

    @property
    def format_bytes(self, image_format='png'):
        """
        convenience but low speed
        :param image_format:
        :return:
        """
        return cv2.imencode(".{}".format(image_format), self.cv_image)[1].tobytes()

    # ===== for preprocess data through cv2 to onnx model =====
    def set_blob(self, input_std, input_mean, input_size):
        self.input_std = input_std
        self.input_mean = input_mean
        self.input_size = input_size
        return self

    @property
    def blob_rgb(self):
        assert self.input_std and self.input_mean and self.input_size, 'Use set_blob first!'
        if not isinstance(self.cv_image, list):
            self.cv_image = [self.cv_image]

        return cv2.dnn.blobFromImages(self.cv_image, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

    def innormal(self, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.
        Args:
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        self.cv_image = np.float32(self.cv_image)
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        assert self.cv_image.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB, self.cv_image)  # inplace
        cv2.subtract(self.cv_image, mean, self.cv_image)  # inplace
        cv2.multiply(self.cv_image, stdinv, self.cv_image)  # inplace
        return self.cv_image


    # ===== convert numpy image to transformed tensor =====
    def set_transform(self, transform=None):
        from torchvision import transforms
        if not transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        return self

    @property
    def tensor(self):
        import torch
        assert self.transform is not None, 'Use set_transform first !'
        img = self.transform(self.cv_image)
        return torch.unsqueeze(img, 0)
