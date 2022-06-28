# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import os
import cv2
import numpy as np
import base64
import io
from pathlib import PosixPath, Path

"""
skimage and pillow read image based uint8 and RGB mode
opencv read image based uint8 and BGR mode
using opencv as the default image read method
"""


class ImageBasic:
    def __init__(self, image_in, image_format, image_size):
        if image_in is None:
            self.cv_image = None
            return
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

    def resize(self, size, interpolation=cv2.INTER_LINEAR):
        if type(size) == tuple:
            if size != self.cv_image.shape[:-1]:
                # cv2 resize function always returns a new Mat object.
                self.cv_image = cv2.resize(self.cv_image, size)
        elif type(size) == int:
            if size != self.cv_image.shape[0]:
                self.cv_image = cv2.resize(self.cv_image, (size, size), interpolation=interpolation)
        else:
            raise 'Check the size input !'
        return self

    def resize_keep_ratio(self, target_size, pad_value=(0, 0, 0)):
        old_size = self.cv_image.shape[0:2][::-1]
        # ratio = min(float(target_size)/(old_size))
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        self.cv_image = cv2.resize(self.cv_image, (new_size[0], new_size[1]))
        pad_w = target_size[0] - new_size[0]
        pad_h = target_size[1] - new_size[1]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        self.cv_image = cv2.copyMakeBorder(self.cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                                           pad_value)
        return self.cv_image, ratio, pad_w, pad_h

    def crop_keep_ratio(self, box, target_size, padding_ratio=1.25, pad_value=(0, 0, 0)):
        """

        Args:
            box: [x1,y1,x2,y2]
            target_size: (w,h)
            pad_value: cv defaule=0

        Returns:

        """
        assert padding_ratio > 1
        image_h, image_w = self.cv_image.shape[:2]

        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        aspect_ratio = target_size[0] / target_size[1]

        if box_w > aspect_ratio * box_h:
            pad_w = box_w * ((padding_ratio - 1) / 2)
            pad_h = ((box_w + pad_w * 2) * 1.0 / aspect_ratio - box_h) / 2
        elif box_w < aspect_ratio * box_h:
            pad_h = box_h * ((padding_ratio - 1) / 2)
            pad_w = ((box_h + pad_h * 2) * aspect_ratio - box_w) / 2

        top, bottom = int(box[1] - pad_h), int(box[3] + pad_h)
        left, right = int(box[0] - pad_w), int((bottom - top) * aspect_ratio + int(box[0] - pad_w))

        border_top = 0
        border_bottom = 0
        border_left = 0
        border_right = 0
        if top < 0:
            border_top = -top
        if left < 0:
            border_left = -left
        if bottom > image_h:
            border_bottom = bottom - image_h
        if right > image_w:
            border_right = right - image_w

        self.cv_image = cv2.copyMakeBorder(self.cv_image, border_top, border_bottom, border_left, border_right,
                                           cv2.BORDER_CONSTANT, None, pad_value)
        self.cv_image = self.cv_image[top:bottom, left:right, :]
        self.cv_image = self.resize(target_size).bgr
        ratio = target_size[1] / (bottom - top)

        return self.cv_image, ratio, left, top

    @staticmethod
    def recover_from_loc(loc, ratio, pad_w, pad_h):
        assert type(loc) == list and len(loc) == 2
        if pad_w == 0:
            y_pad = pad_h // 2
            return [round(loc[0] * 1 / ratio), round((loc[1] - y_pad) * 1 / ratio)]
        if pad_h == 0:
            x_pad = pad_w // 2
            return [round((loc[0] - x_pad) * 1 / ratio), round(loc[1] * 1 / ratio)]

    def show(self, wait_time=0, window_name='test'):
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.cv_image)
        cv2.waitKey(wait_time)
        # key = cv2.waitKey(wait_time) & 0xFF
        # # check for 'q' key-press
        # if key == ord("q"):
        #     # if 'q' key-pressed break out
        #     return False

    def crop_margin(self, box, margin_ratio=0.3):
        """
        :param box: x1,y1,x2,y2
        :param margin_ratio:
        :return:
        """
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        height, width = self.cv_image.shape[0:2]

        margin = int(margin_ratio * box_height)  # if use loose crop, change 0.3 to 1.0

        # new_box = [max(box[0] - margin, 0), max(box[1] - margin, 0),
        #            min(box[2] + margin, width), min(box[3] + margin, height)]
        return self.cv_image[max(box[1] - margin, 0):min(box[3] + margin, height),
               max(box[0] - margin, 0):min(box[2] + margin, width), :]

    def save(self, img_save_p, compress=False, create_path=False):
        if create_path:
            os.makedirs(str(Path(img_save_p).parent), exist_ok=True)
        if not compress:
            cv2.imwrite(img_save_p, self.cv_image)
        else:
            suffix = Path(img_save_p).suffix
            assert suffix not in img_save_p[:-len(suffix)]
            cv2.imwrite(img_save_p.replace(suffix, '.jpg'), self.cv_image)


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
