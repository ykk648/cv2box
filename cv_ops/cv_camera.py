# -- coding: utf-8 --
# @Time : 2022/7/15
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box

import numpy as np
from .cv_file import CVFile
from .cv_image import CVImage
from ..utils import try_import
import cv2


class CVCamera:
    def __init__(self, multical_pkl_path=None, group_pkl_path=None):

        if multical_pkl_path:
            multical = try_import('multical', 'cv_camera: pip install multical')
            self.multical_pkl_data = CVFile(multical_pkl_path).data
            self.cam_name_list = self.multical_pkl_data.calibrations['calibration'].camera_poses.names
        else:
            self.multical_pkl_data = None
        if group_pkl_path:
            self.camera_group = CVFile(group_pkl_path).data
        else:
            self.camera_group = None

    def __len__(self):
        return len(self.cam_name_list)

    @staticmethod
    def matrix_2_rt(matrix_):
        aniposelib = try_import('aniposelib', 'cv_camera: pip install aniposelib')
        r_vec, t_vec = aniposelib.utils.get_rtvec(matrix_)
        return r_vec, t_vec

    @staticmethod
    def rt_2_matrix(r_vec, t_vec):
        """
        Args:
            r_vec: 3,
            t_vec: 3,
        Returns: 4*4
        """
        aniposelib = try_import('aniposelib', 'cv_camera: pip install aniposelib')
        matrix_ = aniposelib.utils.make_M(r_vec, t_vec)
        return matrix_

    @staticmethod
    def multi_cam_stack(results):
        return np.stack([results[key] for key in results.keys()])

    def intri_matrix(self):
        """
        :return: intri 3*3
        """
        intri = {}
        for i in range(len(self.cam_name_list)):
            intri[self.cam_name_list[i]] = self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[
                i].intrinsic
        return intri

    def camera_view_extri_matrix(self):
        camera_m = {}
        for i in range(len(self.cam_name_list)):
            camera_m[self.cam_name_list[i]] = self.multical_pkl_data.calibrations['calibration'].camera_poses[
                                                  self.cam_name_list[i]][:3, 3]
        return camera_m

    def world_view_extri_matrix(self):
        """
        :return: world_m 4*4
        """
        world_m = {}
        trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        for i in range(len(self.cam_name_list)):
            world_m[self.cam_name_list[i]] = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
                                                  self.cam_name_list[i]] @ trans_pose)
        return world_m

    def rt_34(self):
        """
        :return: rt_34 3*4
        """
        world_m = self.world_view_extri_matrix()
        rt_34 = world_m.copy()
        for k, v in rt_34.items():
            rt_34[k] = v[:3]
        return rt_34

    def pall(self):
        """
        :return: pall 3*4
        """
        pall = {}
        intri = self.intri_matrix()
        rt34 = self.rt_34()
        for cam_name in self.cam_name_list:
            k_33 = intri[cam_name]
            rt_34 = rt34[cam_name]
            pall[cam_name] = k_33 @ rt_34
        return pall

    def pall_rotate(self):
        """
        :return: pall rotate by y axis for unity showing
        """
        Z = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]).reshape((4, 4))
        pall = self.pall()
        pall_rotate = {}
        for cam_name in self.cam_name_list:
            pall_rotate[cam_name] = (Z @ pall[cam_name].T).T

        # pall_rotate = {}
        # trans_pose = self.multical_pkl_data.calibrations['calibration'].pose_estimates['times']['poses'][0]
        # for i in range(len(self.cam_name_list)):
        #     # Z = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape((3, 3))
        #     Z = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape((3, 3))
        #     k_33 = self.multical_pkl_data.calibrations['calibration'].cameras.param_objects[i].intrinsic
        #     rt_34 = (self.multical_pkl_data.calibrations['calibration'].camera_poses[
        #                  self.cam_name_list[i]] @ trans_pose)[:3]
        #     r_33 = rt_34[:, :3]
        #     t_13 = rt_34[:, 3].reshape((3, 1))
        #     C = -r_33.T @ t_13
        #     R_new = (Z @ r_33.T).T
        #     Tvec_new = -(R_new) @ (Z @ C)
        #     RT_new = np.hstack((R_new, Tvec_new))
        #     pall_rotate[self.cam_name_list[i]] = k_33 @ RT_new
        return pall_rotate

    def rt_vec(self):
        rt_vec = {}
        world_m = self.world_view_extri_matrix()
        for cam_name in self.cam_name_list:
            r_vec, t_vec = self.matrix_2_rt(world_m[cam_name])
            rt_vec[cam_name] = {'r_vec': r_vec, 't_vec': t_vec}
        return rt_vec

    def dist(self):
        """
        Returns: multical dist: 1*5
        """
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

    # ========================  Board Relate ==============================
    def generate_board(self):
        try:
            dict_id = getattr(cv2.aruco, f'DICT_4X4_250')
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            aruco_dict.bytesList = aruco_dict.bytesList[0:]
        except Exception as e:
            raise 'aruco needs opencv-contrib-python ! '
        charuco_board = cv2.aruco.CharucoBoard_create(7, 5, 0.13, 0.104,
                                                      aruco_dict)
        board_image = charuco_board.draw((500, 500), marginSize=20)
        CVImage(board_image).show()

    def detect_board(self, detect_image, aruco_dict):
        detect_config = cv2.aruco.DetectorParameters_create()

        # setattr(detect_config, 'polygonalApproxAccuracyRate', 0.03)
        # setattr(detect_config, 'adaptiveThreshWinSizeMin', 3)
        # setattr(detect_config, 'adaptiveThreshWinSizeMax', 23)
        # setattr(detect_config, 'adaptiveThreshWinSizeStep', 1)
        # setattr(detect_config, 'adaptiveThreshConstant', 7)
        # setattr(detect_config, 'minMarkerPerimeterRate', 0.01)
        # setattr(detect_config, 'maxMarkerPerimeterRate', 4.0)
        # setattr(detect_config, 'maxErroneousBitsInBorderRate', 0.35)
        # setattr(detect_config, 'errorCorrectionRate', 0.9)
        # setattr(detect_config, 'markerBorderBits', 0.1)
        # setattr(detect_config, 'perspectiveRemovePixelPerCell', 1)
        # setattr(detect_config, 'minOtsuStdDev', 5)
        # setattr(detect_config, 'cornerRefinementMethod', cv2.aruco.CORNER_REFINE_SUBPIX)
        # setattr(detect_config, 'cornerRefinementWinSize', cv2.aruco.CORNER_REFINE_SUBPIX)

        corners, ids, rejected = cv2.aruco.detectMarkers(detect_image,
                                                         aruco_dict, parameters=detect_config)
        cv2.aruco.drawDetectedMarkers(detect_image, corners, ids, )
        CVImage(detect_image).show()

    # ========================  Anipose Format ==============================
    def load_camera_group_from_multical(self):
        """Load a set of cameras in the environment."""
        cameras = []
        for i in range(len(self.cam_name_list)):
            camera_name = self.cam_name_list[i]
            aniposelib = try_import('aniposelib', 'cv_camera: pip install aniposelib')
            camera = aniposelib.cameras.Camera(name=camera_name,
                                               size=self.image_size()[camera_name],
                                               matrix=self.intri_matrix()[camera_name],
                                               rvec=self.rt_vec()[camera_name]['r_vec'],
                                               tvec=self.rt_vec()[camera_name]['t_vec'],
                                               dist=self.dist()[camera_name])
            cameras.append(camera)
        self.camera_group = aniposelib.cameras.CameraGroup(cameras)
        return self

    def bundle_adjust_iter(self, n_view_2d_kps, kpt_thre=0.7):
        """
        :param n_view_2d_kps: N_view * N_frame * N_kps * 3
        :return:
        """
        assert self.camera_group, "use load_camera_group first ! "
        # Filter keypoints to select those best points
        ignore_idxs = np.where(n_view_2d_kps[:, :, :, 2] < kpt_thre)
        n_view_2d_kps[ignore_idxs[0], ignore_idxs[1], ignore_idxs[2], :] = np.nan
        n_view_2d_kps = n_view_2d_kps[..., 0:2]

        # Apply bundle adjustment and dump the camera parameters
        nviews = n_view_2d_kps.shape[0]
        self.camera_group.bundle_adjust_iter(
            n_view_2d_kps.reshape(nviews, -1, 2),
            n_iters=20,
            n_samp_iter=500,
            n_samp_full=5000,
            verbose=True)
        return self.camera_group

    def pall_from_cgroup(self, cgroup_in):
        pall = {}
        for cam in cgroup_in.cameras:
            name = cam.name
            dist = cam.dist
            intri = cam.matrix
            r_vec = cam.rvec
            t_vec = cam.tvec
            matrix_ = self.rt_2_matrix(r_vec, t_vec)
            pall[name] = intri @ matrix_[:3]
        return pall
