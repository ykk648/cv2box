# -- coding: utf-8 --
# @Time : 2022/5/18
# @Author : ykk648
# @Project : https://github.com/ykk648/cv2box
import numpy as np
from ..utils import try_import

try_import('torch', 'cv_rotate: need torch')


# from torch.nn import functional as F
# import cv2
# import numpy.matlib as npm
# import math


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138
def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    # Example:
    #     >>> angle_axis = torch.rand(2, 4)  # Nx4
    #     >>> quaternion = tgm.angle_axis_to_quaternion(angle_axis)  # Nx3
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}"
                         .format(angle_axis.shape))
    # unpack input and compute conversion
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        # >>> quaternion = torch.rand(2, 4)  # Nx4
        # >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


class CVRotate:
    def __init__(self, rotate, rotate_format='angle_axis'):
        # from mathutils import *
        self.rotate = rotate
        self.rotate_format = rotate_format
        # if rotate_format == 'angle_axis'
        # pass

    @staticmethod
    def __angle_axis_to_quaternion_torch(aa):
        aa = aa.clone()
        if aa.dim() == 1:
            assert aa.size(0) == 3
            aa = aa.view(1, 3)
            quat = angle_axis_to_quaternion(aa)[0]
        elif aa.dim() == 2:
            assert aa.size(1) == 3
            quat = angle_axis_to_quaternion(aa)
        else:
            assert aa.dim() == 3
            dim0 = aa.size(0)
            dim1 = aa.size(1)
            assert aa.size(2) == 3
            aa = aa.view(dim0 * dim1, 3)
            quat = angle_axis_to_quaternion(aa)
            quat = quat.view(dim0, dim1, 4)
        return quat

    def angle_axis_to_quaternion(self, angle_axis):
        aa = angle_axis
        if isinstance(aa, torch.Tensor):
            return self.__angle_axis_to_quaternion_torch(aa)
        else:
            assert isinstance(aa, np.ndarray)
            aa_torch = torch.from_numpy(aa)
            quat_torch = self.__angle_axis_to_quaternion_torch(aa_torch)
            return quat_torch.numpy()

    @staticmethod
    def __quaternion_to_angle_axis_torch(quat):
        quat = quat.clone()
        if quat.dim() == 1:
            assert quat.size(0) == 4
            quat = quat.view(1, 4)
            angle_axis = quaternion_to_angle_axis(quat)[0]
        elif quat.dim() == 2:
            assert quat.size(1) == 4
            angle_axis = quaternion_to_angle_axis(quat)
        else:
            assert quat.dim() == 3
            dim0 = quat.size(0)
            dim1 = quat.size(1)
            assert quat.size(2) == 4
            quat = quat.view(dim0 * dim1, 4)
            angle_axis = quaternion_to_angle_axis(quat)
            angle_axis = angle_axis.view(dim0, dim1, 3)
        return angle_axis

    def quaternion_to_angle_axis(self, quaternion):
        quat = quaternion
        if isinstance(quat, torch.Tensor):
            return self.__quaternion_to_angle_axis_torch(quaternion)
        else:
            assert isinstance(quat, np.ndarray)
            quat_torch = torch.from_numpy(quat)
            angle_axis_torch = self.__quaternion_to_angle_axis_torch(quat_torch)
            return angle_axis_torch.numpy()

    def quaternion(self):
        if self.rotate_format == 'angle_axis':
            return self.angle_axis_to_quaternion(self.rotate)

    def angle_axis(self):
        if self.rotate_format == 'quaternion':
            return self.quaternion_to_angle_axis(self.rotate)
