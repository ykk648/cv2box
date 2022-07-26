# -- coding: utf-8 --
# @Time : 2022/7/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

import numpy as np
from cv2box.cv_ops.cv_file import CVFile

try:
    import aniposelib
    from aniposelib.utils import get_rtvec, make_M
except:
    print('cv_camera op need aniposelib !')


class CVCamera:
    def __init__(self, multical_pkl_path=None):
        if multical_pkl_path:
            self.multical_pkl_data = CVFile(multical_pkl_path).data
            self.cam_name_list = self.multical_pkl_data.calibrations['calibration'].camera_poses.names

    @staticmethod
    def matrix_2_rt(matrix_):
        r_vec, t_vec = get_rtvec(matrix_)
        return r_vec, t_vec

    @staticmethod
    def rt_2_matrix(r_vec, t_vec):
        matrix_ = make_M(r_vec, t_vec)
        return matrix_

    @staticmethod
    def multi_cam_stack(results):
        return np.stack([results[key] for key in results.keys()])

    def __len__(self):
        return len(self.cam_name_list)

    def camera_view_extri_matrix(self):
        camera_m = {}
        for i in range(len(self.cam_name_list)):
            camera_m[self.cam_name_list[i]] = self.multical_pkl_data.calibrations['calibration'].camera_poses[
                                                  self.cam_name_list[i]][:3, 3]
        return camera_m

    def world_view_extri_matrix(self):
        world_m = {}
        trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        for i in range(len(self.cam_name_list)):
            world_m[self.cam_name_list[i]] = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
                                                  self.cam_name_list[i]] @ trans_pose)
        return world_m

    def rt_34(self):
        rt_34 = {}
        trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        for i in range(len(self.cam_name_list)):
            rt_34[self.cam_name_list[i]] = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
                                                self.cam_name_list[i]] @ trans_pose)[:3]
        return rt_34

    def pall(self):
        pall = {}
        trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        for i in range(len(self.cam_name_list)):
            k_33 = self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[i].intrinsic
            rt_34 = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
                         self.cam_name_list[i]] @ trans_pose)[:3]
            pall[self.cam_name_list[i]] = k_33 @ rt_34
        return pall

    def pall_rotate(self):
        pall_rotate = {}
        trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        for i in range(len(self.cam_name_list)):
            # Z = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape((3, 3))
            Z = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape((3, 3))
            k_33 = self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[i].intrinsic
            rt_34 = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
                         self.cam_name_list[i]] @ trans_pose)[:3]
            r_33 = rt_34[:, :3]
            t_13 = rt_34[:, 3].reshape((3, 1))
            C = -r_33.T @ t_13
            R_new = (Z @ r_33.T).T
            Tvec_new = -(R_new) @ (Z @ C)
            RT_new = np.hstack((R_new, Tvec_new))
            pall_rotate[self.cam_name_list[i]] = k_33 @ RT_new
        return pall_rotate

    def intri_matrix(self):
        intri = {}
        for i in range(len(self.cam_name_list)):
            intri[self.cam_name_list[i]] = self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[
                i].intrinsic
        return intri

    def rt_vec(self):
        rt_vec = {}
        for i in range(len(self.cam_name_list)):
            r_vec, t_vec = self.matrix_2_rt(self.world_view_extri_matrix()[self.cam_name_list[i]])
            rt_vec[self.cam_name_list[i]] = {'r_vec': r_vec, 't_vec': t_vec}
        return rt_vec

    def dist(self):
        dist = {}
        for i in range(len(self.cam_name_list)):
            dist[self.cam_name_list[i]] = \
                self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[i].dist[0]
        return dist

    def image_size(self):
        image_size = {}
        for i in range(len(self.cam_name_list)):
            image_size[self.cam_name_list[i]] = \
                self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[i].image_size
        return image_size
