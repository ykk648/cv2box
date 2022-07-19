# -- coding: utf-8 --
# @Time : 2022/6/28
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import numpy as np
from cv2box.utils import CalDistance, Normalize


class CVBbox:
    def __init__(self, bbox_list):
        """

        :param bbox_list: [[x1,y1,x2,y2], [...]]
        """
        self.bbox_list = bbox_list

    def area_filter(self, sort_way='big', max_num=1):
        box_area = []
        box_results = []
        aim_ind = None
        box_list_temp = self.bbox_list.copy()
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
        box_list_temp = self.bbox_list.copy()
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
        if len(self.bbox_list) == 0:
            return [[]]
        box_results = []
        dummy1 = []
        dummy2 = []
        box_list_temp = self.bbox_list.copy()
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
