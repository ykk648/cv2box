# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

import numpy as np
from ..utils import CalDistance, Normalize
from ..cv_ops import CVImage


class CVBbox:
    def __init__(self, bbox_array=None):
        """
        :param bbox_array: list or ndarray [[x1,y1,x2,y2], [...]]
        """
        self.bbox_array = bbox_array
        if self.bbox_array is None:
            return
        if not isinstance(self.bbox_array, np.ndarray):
            self.bbox_array = np.array(self.bbox_array)
        if self.bbox_array.ndim != 2:
            self.bbox_array = self.bbox_array[np.newaxis, :]

    def area(self):
        return (self.bbox_array[:, 2] - self.bbox_array[:, 0]) * (self.bbox_array[:, 3] - self.bbox_array[:, 1])

    def __len__(self):
        return self.bbox_array.shape[0]

    def area_filter(self, sort_way='big', max_num=1):
        box_area = []
        box_results = []
        aim_ind = None
        box_list_temp = self.bbox_array.copy()
        for box in box_list_temp:
            box_area.append((box[2] - box[0]) * (box[3] - box[1]))
        for i in range(max_num):
            if sort_way == 'big':
                aim_ind = np.argmax(np.array(box_area))
            elif sort_way == 'small':
                aim_ind = np.argmin(np.array(box_area))
            box_results.append(box_list_temp[aim_ind])
            box_list_temp.pop(aim_ind)
            box_area.pop(aim_ind)
        return box_results

    def center_filter(self, image_shape, sort_way='center', max_num=1):
        box_dis_from_center = []
        box_results = []
        aim_ind = None
        box_list_temp = self.bbox_array.copy()
        for box in box_list_temp:
            box_dis = CalDistance((box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2),
                                  (image_shape[0] / 2, image_shape[1] / 2)).euc()
            box_dis_from_center.append(box_dis)
        for i in range(max_num):
            if sort_way == 'outer':
                aim_ind = np.argmax(np.array(box_dis_from_center))
            elif sort_way == 'center':
                aim_ind = np.argmin(np.array(box_dis_from_center))
            box_results.append(box_list_temp[aim_ind])
            box_list_temp.pop(aim_ind)
            box_dis_from_center.pop(aim_ind)
        return box_results

    def area_center_filter(self, image_shape, max_num=1):
        """

        Args:
            image_shape: HWC
            max_num:

        Returns:

        """
        if len(self.bbox_array) == 0:
            return [[]]
        box_results = []
        dummy1 = []
        dummy2 = []
        if not isinstance(self.bbox_array, list):
            self.bbox_array = self.bbox_array.tolist()
        box_list_temp = self.bbox_array.copy()
        for box in box_list_temp:
            dummy1.append(CalDistance([box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2],
                                      [image_shape[1] / 2, image_shape[0] / 2]).euc())
            dummy2.append((box[2] - box[0]) * (box[3] - box[1]))

        box_area_dis = Normalize(dummy2).np_norm() - Normalize(dummy1).np_norm()
        for i in range(max_num):
            aim_ind = np.argmax(np.array(box_area_dis))

            box_results.append(box_list_temp[aim_ind])
            box_list_temp.pop(aim_ind)
            box_area_dis.tolist().pop(aim_ind)
        return box_results

    @staticmethod
    def get_bbox_from_points(points_in_, image_shape, margin_ratio=0.8):
        """
        :param points_in_:  list [x1,y1,x2,y2...] [[x1,y1],[x2,y2]]
        numpy array N*2 [[x1,y1],[x2,y2]]
        :param image_shape: (W,H)
        :param margin_ratio:
        :return: x1,y1,x2,y2
        """
        points_in_ = CVImage(None).read_points(points_in_)
        min_x = np.min(points_in_[:, 0])
        min_y = np.min(points_in_[:, 1])
        max_x = np.max(points_in_[:, 0])
        max_y = np.max(points_in_[:, 1])
        height = image_shape[1]
        width = image_shape[0]
        margin = int(margin_ratio * max(max_x - min_x, max_y - min_y))  # if use loose crop, change 0.3 to 1.0

        return [int(max(min_x - margin, 0)), int(max(min_y - margin, 0)), int(min(max_x + margin, width)),
                int(min(max_y + margin, height))]
