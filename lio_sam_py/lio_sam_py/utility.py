# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2020 Tixiao Shan
# Copyright 2021 Christoph Gruber
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
# Original File: include/lio_sam/utility.hpp
# Original Author: Tixiao Shan
# Original Source: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
#
# Modifications: This file is a Python port of the original C++ implementation
#                by Eugene Auh in 2026.
# -----------------------------------------------------------------------------
from copy import deepcopy
from enum import Enum
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSLivelinessPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header

POINT_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
    ]
)

POINT_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]

_POINT_FIELD_DTYPES = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


class SensorType(Enum):
    VELODYNE = 0
    OUSTER = 1
    LIVOX = 2
    LIVOX360 = 3


class ParamServer(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.robot_id = ""

        # Topics
        self.point_cloud_topic = self.declare_parameter(
            "pointCloudTopic", "points"
        ).value
        self.imu_topic = self.declare_parameter("imuTopic", "imu/data").value
        self.odom_topic = self.declare_parameter(
            "odomTopic", "lio_sam/odometry/imu"
        ).value
        self.gps_topic = self.declare_parameter(
            "gpsTopic", "lio_sam/odometry/gps"
        ).value

        # Frames
        self.lidar_frame = self.declare_parameter(
            "lidarFrame", "laser_data_frame"
        ).value
        self.baselink_frame = self.declare_parameter("baselinkFrame", "base_link").value
        self.odometry_frame = self.declare_parameter("odometryFrame", "odom").value
        self.map_frame = self.declare_parameter("mapFrame", "map").value

        # GPS Settings
        self.use_imu_heading_initialization = self.declare_parameter(
            "useImuHeadingInitialization", False
        ).value
        self.use_gps_elevation = self.declare_parameter("useGpsElevation", False).value
        self.gps_cov_threshold = self.declare_parameter("gpsCovThreshold", 2.0).value
        self.pose_cov_threshold = self.declare_parameter("poseCovThreshold", 25.0).value

        # Save pcd
        self.save_pcd = self.declare_parameter("savePcd", False).value
        self.save_pcd_directory = self.declare_parameter(
            "savePCDDirectory", "/Downloads/LOAM/"
        ).value

        # Lidar Sensor Configuration
        sensor_str = self.declare_parameter("sensor", "ouster").value
        if sensor_str == "velodyne":
            self.sensor = SensorType.VELODYNE
        elif sensor_str == "ouster":
            self.sensor = SensorType.OUSTER
        elif sensor_str == "livox":
            self.sensor = SensorType.LIVOX
        elif sensor_str == "livox360":
            self.sensor = SensorType.LIVOX360
        else:
            self.get_logger().error(
                "Invalid sensor type "
                "(must be either 'velodyne', 'ouster', 'livox', or 'livox360'): "
                f"{sensor_str}"
            )
            rclpy.shutdown()
        self.n_scan = self.declare_parameter("N_SCAN", 64).value
        self.horizon_scan = self.declare_parameter("Horizon_SCAN", 512).value
        self.downsample_rate = self.declare_parameter("downsampleRate", 1).value
        self.lidar_min_range = self.declare_parameter("lidarMinRange", 5.5).value
        self.lidar_max_range = self.declare_parameter("lidarMaxRange", 1000.0).value

        # IMU
        self.imu_type: Optional[int] = None
        self.imu_acc_noise = self.declare_parameter("imuAccNoise", 9e-4).value
        self.imu_gyr_noise = self.declare_parameter("imuGyrNoise", 1.6e-4).value
        self.imu_acc_bias_n = self.declare_parameter("imuAccBiasN", 5e-4).value
        self.imu_gyr_bias_n = self.declare_parameter("imuGyrBiasN", 7e-5).value
        self.imu_gravity = self.declare_parameter("imuGravity", 9.80511).value
        self.imu_rpy_weight = self.declare_parameter("imuRPYWeight", 0.01).value
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.ext_rot_v = self.declare_parameter("extrinsicRot", identity).value
        self.ext_rpy_v = self.declare_parameter("extrinsicRPY", identity).value
        zero = [0.0, 0.0, 0.0]
        self.ext_trans_v = self.declare_parameter("extrinsicTrans", zero).value
        self.ext_rot = np.array(self.ext_rot_v, dtype=np.float64).reshape(3, 3)
        self.ext_rpy = np.array(self.ext_rpy_v, dtype=np.float64).reshape(3, 3)
        self.ext_trans = np.array(self.ext_trans_v, dtype=np.float64)
        self.ext_qrpy = R.from_matrix(self.ext_rpy).as_quat()

        # LOAM
        self.edge_threshold = self.declare_parameter("edgeThreshold", 1.0).value
        self.surf_threshold = self.declare_parameter("surfThreshold", 0.1).value
        self.edge_feature_min_valid_num = self.declare_parameter(
            "edgeFeatureMinValidNum", 10
        ).value
        self.surf_feature_min_valid_num = self.declare_parameter(
            "surfFeatureMinValidNum", 100
        ).value

        # voxel filter params
        self.odometry_surf_leaf_size = self.declare_parameter(
            "odometrySurfLeafSize", 0.4
        ).value
        self.mapping_corner_leaf_size = self.declare_parameter(
            "mappingCornerLeafSize", 0.2
        ).value
        self.mapping_surf_leaf_size = self.declare_parameter(
            "mappingSurfLeafSize", 0.4
        ).value

        self.z_tollerance = self.declare_parameter("z_tollerance", 1000.0).value
        self.rotation_tollerance = self.declare_parameter(
            "rotation_tollerance", 1000.0
        ).value

        # CPU Params
        self.number_of_cores = self.declare_parameter("numberOfCores", 4).value
        self.mapping_process_interval = self.declare_parameter(
            "mappingProcessInterval", 0.15
        ).value

        # Surrounding map
        self.surroundingkeyframe_adding_dist_threshold = self.declare_parameter(
            "surroundingkeyframeAddingDistThreshold", 1.0
        ).value
        self.surroundingkeyframe_adding_angle_threshold = self.declare_parameter(
            "surroundingkeyframeAddingAngleThreshold", 0.2
        ).value
        self.surrounding_keyframe_density = self.declare_parameter(
            "surroundingKeyframeDensity", 2.0
        ).value
        self.surrounding_keyframe_search_radius = self.declare_parameter(
            "surroundingKeyframeSearchRadius", 50.0
        ).value

        # Loop closure
        self.loop_closure_enable_flag = self.declare_parameter(
            "loopClosureEnableFlag", True
        ).value
        self.loop_closure_frequency = self.declare_parameter(
            "loopClosureFrequency", 1.0
        ).value
        self.surrounding_keyframe_size = self.declare_parameter(
            "surroundingKeyframeSize", 50
        ).value
        self.history_keyframe_search_radius = self.declare_parameter(
            "historyKeyframeSearchRadius", 15.0
        ).value
        self.history_keyframe_search_time_diff = self.declare_parameter(
            "historyKeyframeSearchTimeDiff", 30.0
        ).value
        self.history_keyframe_search_num = self.declare_parameter(
            "historyKeyframeSearchNum", 25
        ).value
        self.history_keyframe_fitness_score = self.declare_parameter(
            "historyKeyframeFitnessScore", 0.3
        ).value

        # global map visualization radius
        self.global_map_visualization_search_radius = self.declare_parameter(
            "globalMapVisualizationSearchRadius", 1000.0
        ).value
        self.global_map_visualization_pose_density = self.declare_parameter(
            "globalMapVisualizationPoseDensity", 10.0
        ).value
        self.global_map_visualization_leaf_size = self.declare_parameter(
            "globalMapVisualizationLeafSize", 1.0
        ).value

        time.sleep(100e-6)

    def imu_converter(self, imu_in: Imu) -> Imu:
        imu_out = deepcopy(imu_in)
        # rotate acceleration
        acc = np.array(
            [
                imu_in.linear_acceleration.x,
                imu_in.linear_acceleration.y,
                imu_in.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        # if self.sensor == SensorType.LIVOX360:
        #     # Livox Mid-360 IMU outputs normzlized acceleration values.
        #     acc *= self.imu_gravity
        acc = self.ext_rot @ acc
        imu_out.linear_acceleration.x = acc[0]
        imu_out.linear_acceleration.y = acc[1]
        imu_out.linear_acceleration.z = acc[2]
        # rotate gyroscope
        gyr = np.array(
            [
                imu_in.angular_velocity.x,
                imu_in.angular_velocity.y,
                imu_in.angular_velocity.z,
            ],
            dtype=np.float64,
        )
        gyr = self.ext_rot @ gyr
        imu_out.angular_velocity.x = gyr[0]
        imu_out.angular_velocity.y = gyr[1]
        imu_out.angular_velocity.z = gyr[2]
        # rotate roll pitch yaw
        q_from = R.from_quat(
            [
                imu_in.orientation.x,
                imu_in.orientation.y,
                imu_in.orientation.z,
                imu_in.orientation.w,
            ]
        )
        rot_final = q_from * R.from_quat(self.ext_qrpy)
        q_final = rot_final.as_quat()
        imu_out.orientation.x = float(q_final[0])
        imu_out.orientation.y = float(q_final[1])
        imu_out.orientation.z = float(q_final[2])
        imu_out.orientation.w = float(q_final[3])

        if np.linalg.norm(q_final) < 0.1:
            self.get_logger().error("Invalid quaternion, please use a 9-axis IMU!")
            rclpy.shutdown()

        return imu_out


def publish_cloud(
    this_pub, this_cloud: np.ndarray, this_stamp, this_frame: str
) -> PointCloud2:
    temp_cloud = _numpy_to_pointcloud2(this_cloud, this_stamp, this_frame)
    if this_pub.get_subscription_count() != 0:
        this_pub.publish(temp_cloud)
    return temp_cloud


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def imu_angular_to_ros_angular(this_imu_msg: Imu) -> tuple[float, float, float]:
    return (
        float(this_imu_msg.angular_velocity.x),
        float(this_imu_msg.angular_velocity.y),
        float(this_imu_msg.angular_velocity.z),
    )


def imu_accel_to_ros_accel(imu_msg: Imu) -> tuple[float, float, float]:
    return (
        float(imu_msg.linear_acceleration.x),
        float(imu_msg.linear_acceleration.y),
        float(imu_msg.linear_acceleration.z),
    )


def imu_rpy_to_ros_rpy(imu_msg: Imu) -> tuple[float, float, float]:
    q = np.array(
        [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w,
        ],
        dtype=np.float64,
    )
    rpy = R.from_quat(q).as_euler("xyz")

    return float(rpy[0]), float(rpy[1]), float(rpy[2])


def point_distance(p: np.ndarray) -> float:
    p_xyz = _point_xyz(p)
    return float(np.linalg.norm(p_xyz))


def points_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    p1_xyz = _point_xyz(p1)
    p2_xyz = _point_xyz(p2)
    return float(np.linalg.norm(p1_xyz - p2_xyz))


def pointcloud2_to_numpy(
    cloud_msg: PointCloud2, np_dtype: np.dtype = POINT_DTYPE
) -> np.ndarray:
    msg_fields = {field.name: field for field in cloud_msg.fields}
    for name in np_dtype.names:
        if name not in msg_fields:
            raise ValueError(f"PointCloud2 message does not contain field '{name}'.")

    use_buffer = True

    if cloud_msg.row_step != cloud_msg.width * cloud_msg.point_step:
        use_buffer = False

    for name in np_dtype.names:
        field = msg_fields[name]

        if field.count != 1:
            use_buffer = False
            break

        dtype_field = np_dtype.fields[name][0]

        if dtype_field.itemsize != _point_field_dtype_size(field.datatype):
            use_buffer = False
            break

    if use_buffer:
        fields = []

        for name in np_dtype.names:
            field = msg_fields[name]

            fields.append(
                (
                    name,
                    _point_field_numpy_dtype(field.datatype, cloud_msg.is_bigendian),
                    field.offset,
                )
            )

        actual_dtype = np.dtype(
            {
                "names": [f[0] for f in fields],
                "formats": [f[1] for f in fields],
                "offsets": [f[2] for f in fields],
                "itemsize": cloud_msg.point_step,
            }
        )
        num_points = cloud_msg.width * cloud_msg.height
        cloud = np.frombuffer(cloud_msg.data, dtype=actual_dtype, count=num_points)

        if actual_dtype == np_dtype:
            return cloud
        else:
            return cloud.astype(np_dtype, copy=False)

    points = point_cloud2.read_points(
        cloud_msg, field_names=np_dtype.names, skip_nans=False
    )
    return np.array(list(points), dtype=np_dtype)


def _numpy_to_pointcloud2(cloud: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    n_points = cloud.shape[0]

    if cloud.dtype.names is not None:
        points = np.ndarray(
            shape=(n_points, 4),
            dtype=np.float32,
            buffer=cloud,
            strides=(cloud.dtype.itemsize, 4),
        )
    else:
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            raise ValueError("Unstructured point cloud must have shape (N, 4).")

        points = np.asarray(cloud, dtype=np.float32)

        if not points.flags.c_contiguous:
            points = np.ascontiguousarray(points)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n_points
    msg.fields = POINT_FIELDS
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n_points
    msg.is_dense = True
    msg.data = points.tobytes()

    return msg


def _point_field_dtype_size(datatype: int) -> int:
    return _POINT_FIELD_DTYPES[datatype].itemsize


def _point_field_numpy_dtype(
    datatype: int,
    bigendian: bool,
) -> np.dtype:
    dtype = _POINT_FIELD_DTYPES[datatype]

    if bigendian:
        return dtype.newbyteorder(">")

    return dtype.newbyteorder("<")


def _point_xyz(p: np.ndarray) -> np.ndarray:
    if getattr(p.dtype, "names", None) is not None:
        return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    else:
        return np.array(p[:3], dtype=np.float64)


qos = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    # deadline=QoSProfile().deadline,
    # lifespan=QoSProfile().lifespan,
    liveliness=QoSLivelinessPolicy.SYSTEM_DEFAULT,
    # liveliness_lease_duration=QoSProfile().liveliness_lease_duration,
    avoid_ros_namespace_conventions=False,
)

qos_imu = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=2000,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    # deadline=QoSProfile().deadline,
    # lifespan=QoSProfile().lifespan,
    liveliness=QoSLivelinessPolicy.SYSTEM_DEFAULT,
    # liveliness_lease_duration=QoSProfile().liveliness_lease_duration,
    avoid_ros_namespace_conventions=False,
)


qos_lidar = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    # deadline=QoSProfile().deadline,
    # lifespan=QoSProfile().lifespan,
    liveliness=QoSLivelinessPolicy.SYSTEM_DEFAULT,
    # liveliness_lease_duration=QoSProfile().liveliness_lease_duration,
    avoid_ros_namespace_conventions=False,
)
