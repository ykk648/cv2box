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
from ..utils import try_import

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
            assert Path(image_in).exists()
            self.cv_image = cv2.imread(image_in)
        elif 'cv2' in image_format:
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
        elif 'gif' in image_format:
            Image = try_import('PIL.Image', 'cv_math: need pillow here.')
            with Image.open(image_in) as im:
                # 获取 GIF 中的每一帧
                frames = []
                for frame in range(im.n_frames):
                    im.seek(frame)
                    frames.append(im.copy())
            self.cv_image = frames
        else:
            raise 'Can not find image_format ！'

    def rgb(self):
        return cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

    def rgba(self):
        return cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGBA)

    def gray(self):
        return cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

    def mask(self, rgb=False):
        """
        Args:
            rgb:
        Returns: (H,W,1)
        """
        if rgb:
            return self.rgb()[:, :, 0][:, :, None]
        else:
            return self.cv_image[:, :, 0][:, :, None]

    @property
    def bgr(self):
        return self.cv_image

    def pillow(self):
        Image = try_import('PIL.Image', 'cv_math: need pillow here.')
        return Image.fromarray(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))

    # ===== for image transfer =====
    def base64(self):
        """
        :return: jpg format base64 code
        """
        image = cv2.imencode('.jpg', self.cv_image)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        return 'data:image/jpg;base64,' + image_code

    def bytes(self):
        """
        fast enough for video real-time steam process
        :return:
        """
        return self.cv_image.tobytes()

    def format_bytes(self, image_format='png'):
        """
        convenience but low speed
        :param image_format:
        :return:
        """
        return cv2.imencode(".{}".format(image_format), self.cv_image)[1].tobytes()

    def resize(self, size, interpolation=cv2.INTER_LINEAR):
        if type(size) == tuple:
            if size != self.cv_image.shape[:-1]:
                # cv2 resize function always returns a new Mat object.
                self.cv_image = cv2.resize(self.cv_image, size, interpolation=interpolation)
        elif type(size) == int:
            if size != self.cv_image.shape[0] or size != self.cv_image.shape[1]:
                self.cv_image = cv2.resize(self.cv_image, (size, size), interpolation=interpolation)
        else:
            raise 'Check the size input !'
        return self

    def show(self, wait_time=0, window_name='test'):
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, self.cv_image)
        cv2.waitKey(wait_time)
        # key = cv2.waitKey(wait_time) & 0xFF
        # # check for 'q' key-press
        # if key == ord("q"):
        #     # if 'q' key-pressed break out
        #     return False

    def save(self, img_save_p, compress=False, create_path=False):
        if create_path:
            os.makedirs(str(Path(img_save_p).parent), exist_ok=True)

        if isinstance(self.cv_image, list):
            # gif to images
            for i, frame in enumerate(self.cv_image):
                save_path = f"{img_save_p}/{i + 1}.png"
                frame.save(save_path, "PNG")
                print(f"Saved frame {i + 1} as {save_path}")
            return

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

    # ===== for image preprocess =====
    def blob(self, input_size, input_mean=127.5, input_std=127.5, rgb=False):
        """
        for preprocess data through cv2 to onnx model
        same as:
            MEAN = 255 * np.array([0.5, 0.5, 0.5])
            STD = 255 * np.array([0.5, 0.5, 0.5])
            x = x.transpose(-1, 0, 1)
            x = (x - MEAN[:, None, None]) / STD[:, None, None]
        Args:
            input_size: (w,h)
            input_mean:
            input_std:
            rgb:
        Returns: 1*3*size*size
        """
        if not isinstance(self.cv_image, list):
            self.cv_image = [self.cv_image]

        return cv2.dnn.blobFromImages(self.cv_image, 1.0 / input_std, input_size,
                                      (input_mean, input_mean, input_mean), swapRB=rgb)

    def blob_innormal(self, input_size, input_mean=[127.5, 127.5, 127.5], input_std=[127.5, 127.5, 127.5], rgb=False,
                      interpolation=cv2.INTER_LINEAR):
        """
        Support 3-channel std/mean, inplace normalize an image with mean and std.
        supprt list/tuple/0-1/1-255
        Args:
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept 0-1/uint8

        self.resize(input_size, interpolation=interpolation)
        if rgb:
            cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB, self.cv_image)  # inplace
        self.cv_image = np.ascontiguousarray(self.cv_image, dtype=np.float32)
        mean = np.float64(np.array(input_mean).reshape(1, -1))
        stdinv = 1 / np.float64(np.array(input_std).reshape(1, -1))
        if input_mean[0] < 1 or input_std[0] < 1:
            mean *= 255
            stdinv /= 255
        cv2.subtract(self.cv_image, mean, self.cv_image)  # inplace
        cv2.multiply(self.cv_image, stdinv, self.cv_image)  # inplace
        return self.cv_image.transpose(2, 0, 1)[np.newaxis,]

    def tensor(self, transform=None):
        """
        convert numpy image to transformed tensor
        Args:
            transform:
        Returns: (1,C,H,W)
        """
        torch = try_import('torch', 'cv_image: need torch here.')
        transforms = try_import('torchvision.transforms', 'cv_image: need torchvision here.')

        if not transform:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        img = transform(self.cv_image)
        return torch.unsqueeze(img, 0)

    def t_normal(self, mean, std, inplace=True):
        """
        Using torchvision transforms , support CHW and BCHW, input tensor
        """
        F = try_import('torchvision.transforms.functional', 'cv_math: need torchvision here.')
        self.cv_image = F.normalize(self.cv_image, mean=mean, std=std, inplace=inplace)

    def t_tensor(self, x, device='cuda'):
        """
        Using torch , support CHW and BCHW
        """
        torch = try_import('torch', 'cv_image: need torch here.')
        if self.cv_image.ndim == 4:
            self.cv_image = torch.from_numpy(x.astype('float32')).permute(3, 1, 2).to(device).div_(255.0)
        else:
            self.cv_image = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).to(device).div_(255.0)

    # ===== for data preprocess and postprocess =====
    def resize_keep_ratio(self, target_size, pad_value=(0, 0, 0)):
        old_size = self.cv_image.shape[0:2][::-1]
        # ratio = min(float(target_size)/(old_size))
        ratio_ = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio_) for i in old_size])
        self.cv_image = cv2.resize(self.cv_image, (new_size[0], new_size[1]))
        pad_w_ = target_size[0] - new_size[0]
        pad_h_ = target_size[1] - new_size[1]
        top, bottom = pad_h_ // 2, pad_h_ - (pad_h_ // 2)
        left, right = pad_w_ // 2, pad_w_ - (pad_w_ // 2)
        self.cv_image = cv2.copyMakeBorder(self.cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                                           pad_value)
        return self.cv_image, ratio_, pad_w_, pad_h_

    @staticmethod
    def recover_from_resize(loc, ratio, pad_w, pad_h):
        """
        reverse method of resize_keep_ratio
        Args:
            loc: N*M M>2
            ratio:
            pad_w:
            pad_h:
        Returns:
        """
        loc = np.array(loc)
        if pad_w == 0:
            y_pad = pad_h // 2
            loc[:, 0] = np.round(loc[:, 0] * 1 / ratio)
            loc[:, 1] = np.round((loc[:, 1] - y_pad) * 1 / ratio)
            return loc
        if pad_h == 0:
            x_pad = pad_w // 2
            loc[:, 0] = np.round((loc[:, 0] - x_pad) * 1 / ratio)
            loc[:, 1] = np.round(loc[:, 1] * 1 / ratio)
            return loc

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

        return self.cv_image[max(box[1] - margin, 0):min(box[3] + margin, height),
               max(box[0] - margin, 0):min(box[2] + margin, width), :]

    def crop_keep_ratio(self, box, target_size, padding_ratio=1.25, pad_value=(0, 0, 0)):
        """

        Args:
            box: [x1,y1,x2,y2]
            target_size: (w,h)
            padding_ratio:
            pad_value: cv defaule=0

        Returns:

        """
        assert padding_ratio >= 1
        image_h, image_w = self.cv_image.shape[:2]

        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        aspect_ratio = target_size[0] / target_size[1]

        # 对四周进行padding
        if box_w > aspect_ratio * box_h:
            pad_w = box_w * ((padding_ratio - 1) / 2)
            pad_h = ((box_w + pad_w * 2) * 1.0 / aspect_ratio - box_h) / 2
        else:
            pad_h = box_h * ((padding_ratio - 1) / 2)
            pad_w = ((box_h + pad_h * 2) * aspect_ratio - box_w) / 2
        top, bottom = int(box[1] - pad_h), int(box[3] + pad_h)
        left, right = int(box[0] - pad_w), int((bottom - top) * aspect_ratio + int(box[0] - pad_w))

        # 旧坐标系下依据padding结果扩充边界
        border_top = 0
        border_bottom = 0
        border_left = 0
        border_right = 0
        if bottom > image_h:
            border_bottom = bottom - image_h
        if right > image_w:
            border_right = right - image_w
        if top < 0:
            border_top = -top
            bottom += -top  # 新坐标系bottom
        if left < 0:
            border_left = -left
            right += -left  # 新坐标系right
        self.cv_image = cv2.copyMakeBorder(self.cv_image, border_top, border_bottom, border_left, border_right,
                                           cv2.BORDER_CONSTANT, None, pad_value)
        # 新坐标系下，top和left不存在负数
        self.cv_image = self.cv_image[max(top, 0):bottom, max(left, 0):right, :]
        self.cv_image = self.resize(target_size).bgr
        # 假如top为负，bottom已经加过top
        ratio = target_size[1] / (bottom - max(top, 0))

        # 保留旧坐标系下的left top用于坐标还原
        return self.cv_image, ratio, left, top

    @staticmethod
    def recover_from_crop(loc, ratio, left, top, image_shape_):
        """
        reverse method of crop_keep_ratio
        Args:
            loc: N*M M>=2   [[x1,y1,..],[x2,y2,..]]
            ratio:
            left:
            top:
            image_shape_: HWC
        Returns: N*M M>=2   [[x1,y1,..],[x2,y2,..]]
        """
        loc = np.array(loc)
        loc[:, 0] = np.round(loc[:, 0] * image_shape_[0] / ratio + left)
        loc[:, 1] = np.round(loc[:, 1] * image_shape_[1] / ratio + top)
        return loc

    @staticmethod
    def recover_from_reverse_matrix(img_fg, img_bg, mat_rev, img_fg_mask=None, blur=False):
        """
        Args:
            img_fg: 0-1
            img_bg:
            mat_rev:
            img_fg_mask: 0-1 (h,w,1) , None means use common mask(only for ROI image
        Returns:
        """
        ne = try_import('numexpr', 'cv_image: this func need numexpr')
        # ne.set_num_threads(1)
        img_fg_trans = cv2.warpAffine(img_fg, mat_rev, img_bg.shape[:2][::-1], borderMode=cv2.BORDER_REPLICATE)

        if img_fg_mask is None:
            from ..utils.util import common_face_mask
            # from cv2box import MyFpsCounter
            # with MyFpsCounter() as mfs:
            # img_fg_mask = mat2mask(img_bg, mat_rev)
            img_fg_mask = common_face_mask(img_bg.shape[:2][::-1])
        else:
            img_fg_mask = cv2.warpAffine(img_fg_mask, mat_rev, img_bg.shape[:2][::-1], borderValue=0)[
                ..., np.newaxis]

        if blur:
            img_fg_mask = cv2.stackBlur(img_fg_mask, (201, 201))[..., np.newaxis]
        # clip
        # img_fg_mask[img_fg_mask > 0.2] = 1

        local_dict = {
            'img_fg_mask': img_fg_mask,
            'img_fg_trans': img_fg_trans,
            'img_bg': img_bg,
        }
        img = ne.evaluate('img_fg_mask * (img_fg_trans * 255)+(1 - img_fg_mask) * img_bg', local_dict=local_dict,
                          global_dict=None)
        img = img.astype(np.uint8)
        return img

    # ===== for points read and draw =====
    @staticmethod
    def read_points(landmark_in_):
        """
        :param landmark_in_: list [x1,y1,x2,y2...] [[x1,y1],[x2,y2]]
        numpy array N*2 [[x1,y1],[x2,y2]]
        :return:numpy array N*2 [[x1,y1],[x2,y2]]
        """
        if isinstance(landmark_in_, list):
            if isinstance(landmark_in_[0], list):
                landmark_in_ = np.array(landmark_in_)
            else:
                assert len(landmark_in_) % 2 == 0
                landmark_in_ = np.array(landmark_in_).reshape((-1, 2))
        return landmark_in_

    def draw_landmarks(self, landmark_in_, color=(0, 255, 0)):
        """
        :param landmark_in_: list [x1,y1,x2,y2...] [[x1,y1],[x2,y2]]
        numpy array N*2 [[x1,y1],[x2,y2]]
        :return:
        """
        landmark_in_ = self.read_points(landmark_in_)
        image_copy = self.cv_image.copy()
        cycle_line = int(self.cv_image.shape[0] / 100)
        for i in range(landmark_in_.shape[0]):
            try:
                cv2.circle(image_copy, (int(landmark_in_[i][0]), int(landmark_in_[i][1])), cycle_line, color=color)
            except ValueError:
                pass
        return image_copy
