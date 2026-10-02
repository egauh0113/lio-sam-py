# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from geometry_msgs.msg import Pose, Transform
from nav_msgs.msg import Odometry


def voxel_downsample(cloud: np.ndarray, leaf_size: float) -> np.ndarray:
    if cloud.size == 0:
        return np.empty(0, dtype=cloud.dtype)
    if leaf_size <= 0.0:
        return cloud.copy()

    xyz = xyz_array(cloud)
    keys = np.floor(xyz / float(leaf_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    out = np.empty(counts.size, dtype=cloud.dtype)
    for name in cloud.dtype.names:
        if name in ("x", "y", "z", "intensity"):
            out[name] = np.bincount(inverse, weights=cloud[name]) / counts
        elif name == "time":
            out[name] = np.bincount(inverse, weights=cloud[name]) / counts
        else:
            out[name] = np.bincount(inverse, weights=cloud[name]) / counts
    return out


# Transform Fusion.
def matrix_to_pose(T: np.ndarray, pose: Pose) -> None:
    trans = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()

    pose.position.x = float(trans[0])
    pose.position.y = float(trans[1])
    pose.position.z = float(trans[2])

    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])


def matrix_to_transform(T: np.ndarray, transform: Transform) -> None:
    trans = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()

    transform.translation.x = float(trans[0])
    transform.translation.y = float(trans[1])
    transform.translation.z = float(trans[2])

    transform.rotation.x = float(quat[0])
    transform.rotation.y = float(quat[1])
    transform.rotation.z = float(quat[2])
    transform.rotation.w = float(quat[3])


def transform_to_matrix(transform: Transform) -> np.ndarray:
    trans = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )
    quat = np.array(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ],
        dtype=np.float64,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = trans

    return T


# IMU Preintegration.
def odom_to_pose3(odom: Odometry) -> gtsam.Pose3:
    rot = gtsam.Rot3.Quaternion(
        float(odom.pose.pose.orientation.w),
        float(odom.pose.pose.orientation.x),
        float(odom.pose.pose.orientation.y),
        float(odom.pose.pose.orientation.z),
    )
    trans = gtsam.Point3(
        float(odom.pose.pose.position.x),
        float(odom.pose.pose.position.y),
        float(odom.pose.pose.position.z),
    )

    return gtsam.Pose3(rot, trans)


# Map Optimization.
def matrix_from_rpy_xyz(
    roll: float, pitch: float, yaw: float, x: float, y: float, z: float
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = [x, y, z]
    return mat


def matrix_to_xyz_rpy(
    mat: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    x, y, z = mat[:3, 3]
    roll, pitch, yaw = R.from_matrix(mat[:3, :3]).as_euler("xyz")
    return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)


def quaternion_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return R.from_euler("xyz", [roll, pitch, yaw]).as_quat()


def slerp_single_axis(current: float, imu: float, axis: str, weight: float) -> float:
    if axis == "roll":
        r0 = R.from_euler("xyz", [current, 0.0, 0.0])
        r1 = R.from_euler("xyz", [imu, 0.0, 0.0])
        idx = 0
    else:
        r0 = R.from_euler("xyz", [0.0, current, 0.0])
        r1 = R.from_euler("xyz", [0.0, imu, 0.0])
        idx = 1
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return float(slerp([weight]).as_euler("xyz")[0, idx])


def xyz_array(cloud: np.ndarray) -> np.ndarray:
    if cloud.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
        np.float64, copy=False
    )
