# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2020 Tixiao Shan
# Copyright 2021 Christoph Gruber
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
# Original File: src/mapOptimization.cpp
# Original Author: Tixiao Shan
# Original Source: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
#
# Modifications: This file is a Python port of the original C++ implementation
#                by Eugene Auh in 2026.
# -----------------------------------------------------------------------------
from collections import deque
from copy import deepcopy
import os
import shutil
import threading
from typing import Optional

import gtsam
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import rclpy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from lio_sam_py.transform_utils import (
    matrix_from_rpy_xyz,
    matrix_to_xyz_rpy,
    quaternion_xyzw_from_rpy,
    slerp_single_axis,
    voxel_downsample,
    xyz_array,
)
from lio_sam_py.utility import (
    POINT_DTYPE,
    SensorType,
    ParamServer,
    points_distance,
    pointcloud2_to_numpy,
    publish_cloud,
    stamp_to_sec,
    qos,
)
from lio_sam_msgs.msg import CloudInfo
from lio_sam_msgs.srv import SaveMap

KEY_POSE_3D_DTYPE = np.dtype(
    [
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("intensity", np.float64),
    ]
)

# A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
POINT_POSE_DTYPE = np.dtype(
    [
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("intensity", np.float64),
        ("roll", np.float64),
        ("pitch", np.float64),
        ("yaw", np.float64),
        ("time", np.float64),
    ]
)


class MapOptimization(ParamServer):
    def __init__(self):
        super().__init__(node_name="map_optimization")

        # gtsam
        self.gtsam_graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.optimized_estimate = gtsam.Values()
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)
        self.isam_current_estimate = gtsam.Values()
        self.pose_covariance = np.zeros((6, 6), dtype=np.float64)

        self.pub_laser_cloud_surround = self.create_publisher(
            PointCloud2, "lio_sam/mapping/map_global", 1
        )
        self.pub_laser_odometry_global = self.create_publisher(
            Odometry, "lio_sam/mapping/odometry", qos
        )
        self.pub_laser_odometry_incremental = self.create_publisher(
            Odometry, "lio_sam/mapping/odometry_incremental", qos
        )
        self.pub_key_poses = self.create_publisher(
            PointCloud2, "lio_sam/mapping/trajectory", 1
        )
        self.pub_path = self.create_publisher(Path, "lio_sam/mapping/path", 1)

        self.pub_history_key_frames = self.create_publisher(
            PointCloud2, "lio_sam/mapping/icp_loop_closure_history_cloud", 1
        )
        self.pub_icp_key_frames = self.create_publisher(
            PointCloud2, "lio_sam/mapping/icp_loop_closure_corrected_cloud", 1
        )
        self.pub_recent_key_frames = self.create_publisher(
            PointCloud2, "lio_sam/mapping/map_local", 1
        )
        self.pub_recent_key_frame = self.create_publisher(
            PointCloud2, "lio_sam/mapping/cloud_registered", 1
        )
        self.pub_cloud_registered_raw = self.create_publisher(
            PointCloud2, "lio_sam/mapping/cloud_registered_raw", 1
        )
        self.pub_loop_constraint_edge = self.create_publisher(
            MarkerArray, "/lio_sam/mapping/loop_closure_constraints", 1
        )

        self.srv_save_map = self.create_service(
            SaveMap, "lio_sam/save_map", self.save_map_service
        )
        self.sub_cloud = self.create_subscription(
            CloudInfo, "lio_sam/feature/cloud_info", self.laser_cloud_info_handler, qos
        )
        self.sub_gps = self.create_subscription(
            Odometry, self.gps_topic, self.gps_handler, 200
        )
        self.sub_loop = self.create_subscription(
            Float64MultiArray,
            "lio_loop/loop_closure_detection",
            self.loop_info_handler,
            qos,
        )

        self.gps_queue = deque()
        self.cloud_info = CloudInfo()

        self.corner_cloud_key_frames: list[np.ndarray] = []
        self.surf_cloud_key_frames: list[np.ndarray] = []

        self.cloud_key_poses_3d = np.empty(0, dtype=KEY_POSE_3D_DTYPE)
        self.cloud_key_poses_6d = np.empty(0, dtype=POINT_POSE_DTYPE)
        self.copy_cloud_key_poses_3d = np.empty(0, dtype=KEY_POSE_3D_DTYPE)
        self.copy_cloud_key_poses_6d = np.empty(0, dtype=POINT_POSE_DTYPE)

        self.laser_cloud_corner_last = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_last = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_corner_last_ds = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_last_ds = np.empty(0, dtype=POINT_DTYPE)

        self.laser_cloud_ori = np.empty(0, dtype=POINT_DTYPE)
        self.coeff_sel = np.empty(0, dtype=POINT_DTYPE)

        scan_size = self.n_scan * self.horizon_scan
        self.laser_cloud_ori_corner_vec = np.empty(
            scan_size, dtype=POINT_DTYPE
        )  # corner point holder for parallel computation
        self.coeff_sel_corner_vec = np.empty(scan_size, dtype=POINT_DTYPE)
        self.laser_cloud_ori_corner_flag = np.zeros(scan_size, dtype=bool)
        self.laser_cloud_ori_surf_vec = np.empty(
            scan_size, dtype=POINT_DTYPE
        )  # surf point holder for parallel computation
        self.coeff_sel_surf_vec = np.empty(scan_size, dtype=POINT_DTYPE)
        self.laser_cloud_ori_surf_flag = np.zeros(scan_size, dtype=bool)

        self.laser_cloud_map_container: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.laser_cloud_corner_from_map = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_from_map = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_corner_from_map_ds = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_from_map_ds = np.empty(0, dtype=POINT_DTYPE)

        self.kdtree_corner_from_map: Optional[cKDTree] = None
        self.kdtree_surf_from_map: Optional[cKDTree] = None

        self.time_laser_info_stamp = None
        self.time_laser_info_cur = -1.0
        self.time_last_processing = -1.0

        self.transform_tobe_mapped = np.zeros(6, dtype=np.float64)

        self.mtx = threading.Lock()
        self.mtx_loop_info = threading.Lock()

        self.is_degenerate = False
        self.mat_P = np.zeros((6, 6), dtype=np.float64)

        self.laser_cloud_corner_from_map_ds_num = 0
        self.laser_cloud_surf_from_map_ds_num = 0
        self.laser_cloud_corner_last_ds_num = 0
        self.laser_cloud_surf_last_ds_num = 0

        self.a_loop_is_closed = False
        self.loop_index_container: dict[int, int] = {}  # from new to old
        self.loop_index_queue: list[tuple[int, int]] = []
        self.loop_pose_queue: list[gtsam.Pose3] = []
        self.loop_noise_queue = []
        self.loop_info_vec = deque()

        self.global_path = Path()

        self.trans_point_associate = np.eye(4, dtype=np.float64)
        self.incremental_odometry_affine_front = np.eye(4, dtype=np.float64)
        self.incremental_odometry_affine_back = np.eye(4, dtype=np.float64)

        self.br = tf2_ros.TransformBroadcaster(self)

        self.last_imu_transformation = np.eye(4, dtype=np.float64)
        self.last_imu_pre_transformation = np.eye(4, dtype=np.float64)
        self.last_imu_pre_trans_available = False
        self.last_gps_point = np.zeros(3, dtype=np.float64)
        self.last_incre_odom_pub_flag = False
        self.incre_odom_affine = np.eye(4, dtype=np.float64)
        self.laser_odom_incremental = Odometry()

    def save_map_service(self, req: SaveMap.Request, res: SaveMap.Response):
        dest = os.path.expanduser("~") + (
            self.save_pcd_directory if not req.destination else str(req.destination)
        )
        self.get_logger().info(f"Saving map to: {dest}")
        res.success = self.save_map_to_directory(dest, float(req.resolution))
        return res

    def laser_cloud_info_handler(self, msg_in: CloudInfo) -> None:
        # extract time stamp
        self.time_laser_info_stamp = msg_in.header.stamp
        self.time_laser_info_cur = stamp_to_sec(self.time_laser_info_stamp)

        # extract info and feature cloud
        self.cloud_info = deepcopy(msg_in)
        self.laser_cloud_corner_last = pointcloud2_to_numpy(msg_in.cloud_corner)
        self.laser_cloud_surf_last = pointcloud2_to_numpy(msg_in.cloud_surface)

        with self.mtx:
            if (
                self.time_laser_info_cur - self.time_last_processing
                >= self.mapping_process_interval
            ):
                self.time_last_processing = self.time_laser_info_cur

                self.update_initial_guess()

                self.extract_surrounding_key_frames()

                self.downsample_current_scan()

                self.scan_to_map_optimization()

                self.save_key_frames_and_factor()

                self.correct_poses()

                self.publish_odometry()

                self.publish_frames()

    def gps_handler(self, gps_msg: Odometry) -> None:
        self.gps_queue.append(gps_msg)

    def point_associate_to_map(self, point) -> np.ndarray:
        p = np.array([point["x"], point["y"], point["z"]], dtype=np.float64)
        return (
            self.trans_point_associate_to_map[:3, :3] @ p
            + self.trans_point_associate_to_map[:3, 3]
        )

    def transform_point_cloud(self, cloud_in: np.ndarray, transform_in) -> np.ndarray:
        if cloud_in.size == 0:
            return np.empty(0, dtype=cloud_in.dtype)

        mat = MapOptimization.point_to_mat(transform_in)
        xyz = xyz_array(cloud_in)
        xyz_out = (mat[:3, :3] @ xyz.T).T + mat[:3, 3]

        cloud_out = cloud_in.copy().astype(POINT_DTYPE, copy=False)
        cloud_out["x"] = xyz_out[:, 0]
        cloud_out["y"] = xyz_out[:, 1]
        cloud_out["z"] = xyz_out[:, 2]

        return cloud_out

    @staticmethod
    def point_to_pose3(p) -> gtsam.Pose3:
        return gtsam.Pose3(
            gtsam.Rot3.RzRyRx(float(p["roll"]), float(p["pitch"]), float(p["yaw"])),
            gtsam.Point3(float(p["x"]), float(p["y"]), float(p["z"])),
        )

    @staticmethod
    def trans_to_pose3(t: np.ndarray) -> gtsam.Pose3:
        return gtsam.Pose3(
            gtsam.Rot3.RzRyRx(float(t[0]), float(t[1]), float(t[2])),
            gtsam.Point3(float(t[3]), float(t[4]), float(t[5])),
        )

    @staticmethod
    def point_to_mat(p) -> np.ndarray:
        return matrix_from_rpy_xyz(
            p["roll"], p["pitch"], p["yaw"], p["x"], p["y"], p["z"]
        )

    @staticmethod
    def trans_to_mat(t: np.ndarray) -> np.ndarray:
        return matrix_from_rpy_xyz(t[0], t[1], t[2], t[3], t[4], t[5])

    @staticmethod
    def trans_to_point_pose(t: np.ndarray) -> np.ndarray:
        p = np.zeros((), dtype=POINT_POSE_DTYPE)
        p["x"] = t[3]
        p["y"] = t[4]
        p["z"] = t[5]
        p["roll"] = t[0]
        p["pitch"] = t[1]
        p["yaw"] = t[2]
        return p

    def visualize_global_map_thread(self) -> None:
        rate = self.create_rate(0.2)
        while rclpy.ok():
            rate.sleep()
            self.publish_global_map()
        if self.save_pcd:
            dest = os.path.expanduser("~") + self.save_pcd_directory
            self.save_map_to_directory(dest, 0.0)

    def publish_global_map(self) -> None:
        if self.pub_laser_cloud_surround.get_subscription_count() == 0:
            return

        if self.cloud_key_poses_3d.shape[0] == 0:
            return

        with self.mtx:
            poses_3d = self.cloud_key_poses_3d.copy()
            poses_6d = self.cloud_key_poses_6d.copy()
        poses_3d_xyz = xyz_array(poses_3d)

        # kd-tree to find near key frames to visualize
        kd_tree_global_map = cKDTree(poses_3d_xyz)
        # search near key frames to visualize
        point_search_ind_global_map = kd_tree_global_map.query_ball_point(
            poses_3d_xyz[-1], self.global_map_visualization_search_radius
        )

        global_map_key_poses = poses_3d[
            np.asarray(point_search_ind_global_map, dtype=np.int64)
        ]
        # downsample near selected key frames
        global_map_key_poses_ds = voxel_downsample(
            global_map_key_poses, self.global_map_visualization_pose_density
        )
        if global_map_key_poses_ds.size:
            _, nn = kd_tree_global_map.query(xyz_array(global_map_key_poses_ds), k=1)
            global_map_key_poses_ds["intensity"] = poses_3d["intensity"][nn]
        parts = []

        # extract visualized and downsampled key frames
        for p in global_map_key_poses_ds:
            if (
                points_distance(p, poses_3d_xyz[-1])
                > self.global_map_visualization_search_radius
            ):
                continue
            this_key_ind = int(round(p["intensity"]))
            parts += [
                self.transform_point_cloud(
                    self.corner_cloud_key_frames[this_key_ind], poses_6d[this_key_ind]
                ),
                self.transform_point_cloud(
                    self.surf_cloud_key_frames[this_key_ind], poses_6d[this_key_ind]
                ),
            ]
        # downsample visuzlied points
        # for global map visualization
        global_map_key_frames = (
            np.concatenate(parts) if parts else np.empty(0, dtype=POINT_DTYPE)
        )
        global_map_key_frames_ds = voxel_downsample(
            global_map_key_frames, self.global_map_visualization_leaf_size
        )

        publish_cloud(
            self.pub_laser_cloud_surround,
            global_map_key_frames_ds,
            self.time_laser_info_stamp,
            self.odometry_frame,
        )

    def loop_closure_thread(self) -> None:
        if not self.loop_closure_enable_flag:
            return

        rate = self.create_rate(self.loop_closure_frequency)
        while rclpy.ok():
            rate.sleep()
            self.perform_loop_closure()
            self.visualize_loop_closure()

    def loop_info_handler(self, loop_msg: Float64MultiArray) -> None:
        with self.mtx_loop_info:
            if len(loop_msg.data) != 2:
                return

            self.loop_info_vec.append(loop_msg)

            while len(self.loop_info_vec) > 5:
                self.loop_info_vec.popleft()

    def perform_loop_closure(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return

        with self.mtx:
            self.copy_cloud_key_poses_3d = self.cloud_key_poses_3d.copy()
            self.copy_cloud_key_poses_6d = self.cloud_key_poses_6d.copy()

        # find keys
        loop_pair = self.detect_loop_closure_external()
        if loop_pair is None:
            loop_pair = self.detect_loop_closure_distance()
        if loop_pair is None:
            return
        loop_key_cur, loop_key_pre = loop_pair

        # extract cloud
        cure_keyframe_cloud = self.loop_find_near_keyframes(loop_key_cur, 0)
        prev_keyframe_cloud = self.loop_find_near_keyframes(
            loop_key_pre, self.history_keyframe_search_num
        )
        if cure_keyframe_cloud.shape[0] < 300 or prev_keyframe_cloud.shape[0] < 1000:
            return
        if self.pub_history_key_frames.get_subscription_count() != 0:
            publish_cloud(
                self.pub_history_key_frames,
                prev_keyframe_cloud,
                self.time_laser_info_stamp,
                self.odometry_frame,
            )

        # ICP Settings
        # Align clouds
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(xyz_array(cure_keyframe_cloud))
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(xyz_array(prev_keyframe_cloud))
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            self.history_keyframe_search_radius * 2.0,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6
            ),
        )

        # publish corrected cloud
        correction = np.asarray(result.transformation, dtype=np.float64)
        if self.pub_icp_key_frames.get_subscription_count() != 0:
            xyz = xyz_array(cure_keyframe_cloud)
            xyz = (correction[:3, :3] @ xyz.T).T + correction[:3, 3]
            closed_cloud = cure_keyframe_cloud.copy()
            closed_cloud["x"] = xyz[:, 0]
            closed_cloud["y"] = xyz[:, 1]
            closed_cloud["z"] = xyz[:, 2]
            publish_cloud(
                self.pub_icp_key_frames,
                closed_cloud,
                self.time_laser_info_stamp,
                self.odometry_frame,
            )

        # Get pose transformation
        t_correct = correction @ MapOptimization.point_to_mat(
            self.copy_cloud_key_poses_6d[loop_key_cur]
        )
        x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(t_correct)
        pose_from = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(roll, pitch, yaw),
            gtsam.Point3(x, y, z),
        )
        pose_to = MapOptimization.point_to_pose3(
            self.copy_cloud_key_poses_6d[loop_key_pre]
        )
        noise_score = max(float(result.inlier_rmse) ** 2, 1e-12)
        if result.fitness <= 0.0 or noise_score > self.history_keyframe_fitness_score:
            return
        constraint_noise = gtsam.noiseModel.Diagonal.Variances(
            np.full(6, noise_score, dtype=np.float64)
        )

        # Add pose constraint
        with self.mtx:
            self.loop_index_queue.append((loop_key_cur, loop_key_pre))
            self.loop_pose_queue.append(pose_from.between(pose_to))
            self.loop_noise_queue.append(constraint_noise)

        # add loop constraint
        self.loop_index_container[loop_key_cur] = loop_key_pre

    def detect_loop_closure_distance(self) -> Optional[tuple[int, int]]:
        loop_key_cur = self.copy_cloud_key_poses_3d.size - 1
        loop_key_pre = -1

        # check loop constraint added before
        if loop_key_cur in self.loop_index_container:
            return None

        # find the closest history key frame
        copy_cloud_key_poses_3d_xyz = xyz_array(self.copy_cloud_key_poses_3d)
        kdtree_history_key_poses = cKDTree(copy_cloud_key_poses_3d_xyz)
        ids = kdtree_history_key_poses.query_ball_point(
            copy_cloud_key_poses_3d_xyz[-1],
            self.history_keyframe_search_radius,
        )

        for id in ids:
            if (
                abs(self.copy_cloud_key_poses_6d[id]["time"] - self.time_laser_info_cur)
                > self.history_keyframe_search_time_diff
            ):
                loop_key_pre = int(id)
                break

        if loop_key_pre == -1 or loop_key_pre == loop_key_cur:
            return None

        return loop_key_cur, loop_key_pre

    def detect_loop_closure_external(self) -> Optional[tuple[int, int]]:
        with self.mtx_loop_info:
            if not self.loop_info_vec:
                return None

            loop_time_cur = self.loop_info_vec[0].data[0]
            loop_time_pre = self.loop_info_vec[0].data[1]
            self.loop_info_vec.popleft()

            if (
                abs(loop_time_cur - loop_time_pre)
                < self.history_keyframe_search_time_diff
            ):
                return None

            cloud_size = self.copy_cloud_key_poses_6d.shape[0]
            if cloud_size < 2:
                return None

            # latest key
            loop_key_cur = cloud_size - 1
            for i in range(cloud_size - 1, -1, -1):
                if self.copy_cloud_key_poses_6d[i]["time"] >= loop_time_cur:
                    loop_key_cur = int(
                        round(self.copy_cloud_key_poses_6d[i]["intensity"])
                    )
                else:
                    break

            # previous key
            loop_key_pre = 0
            for i in range(0, cloud_size):
                if self.copy_cloud_key_poses_6d[i]["time"] <= loop_time_pre:
                    loop_key_pre = int(
                        round(self.copy_cloud_key_poses_6d[i]["intensity"])
                    )
                else:
                    break

            if loop_key_pre == loop_key_cur:
                return None

            if loop_key_cur in self.loop_index_container:
                return None

            return loop_key_cur, loop_key_pre

    def loop_find_near_keyframes(self, key: int, search_num: int) -> np.ndarray:
        # extract near keyframes
        near_keyframes = []
        cloud_size = self.copy_cloud_key_poses_6d.shape[0]
        for i in range(-search_num, search_num + 1):
            key_near = key + i
            if key_near < 0 or key_near >= cloud_size:
                continue
            near_keyframes.append(
                self.transform_point_cloud(
                    self.corner_cloud_key_frames[key_near],
                    self.copy_cloud_key_poses_6d[key_near],
                )
            )
            near_keyframes.append(
                self.transform_point_cloud(
                    self.surf_cloud_key_frames[key_near],
                    self.copy_cloud_key_poses_6d[key_near],
                )
            )

        if not near_keyframes:
            return np.empty(0, dtype=POINT_DTYPE)

        # downsample near keyframes
        return voxel_downsample(
            np.concatenate(near_keyframes), self.mapping_surf_leaf_size
        )

    def visualize_loop_closure(self) -> None:
        if not self.loop_index_container:
            return

        marker_array = MarkerArray()
        # loop nodes
        marker_node = Marker()
        marker_node.header.frame_id = self.odometry_frame
        marker_node.header.stamp = self.time_laser_info_stamp
        marker_node.action = Marker.ADD
        marker_node.type = Marker.SPHERE_LIST
        marker_node.ns = "loop_nodes"
        marker_node.id = 0
        marker_node.pose.orientation.w = 1.0
        marker_node.scale.x = 0.3
        marker_node.scale.y = 0.3
        marker_node.scale.z = 0.3
        marker_node.color.r = 0.0
        marker_node.color.g = 0.8
        marker_node.color.b = 1.0
        marker_node.color.a = 1.0
        # loop edges
        marker_edge = Marker()
        marker_edge.header.frame_id = self.odometry_frame
        marker_edge.header.stamp = self.time_laser_info_stamp
        marker_edge.action = Marker.ADD
        marker_edge.type = Marker.LINE_LIST
        marker_edge.ns = "loop_edges"
        marker_edge.id = 1
        marker_edge.pose.orientation.w = 1.0
        marker_edge.scale.x = 0.1
        marker_edge.color.r = 0.9
        marker_edge.color.g = 0.9
        marker_edge.color.b = 0.0
        marker_edge.color.a = 1.0

        for key_cur, key_pre in self.loop_index_container.items():
            p = Point()
            p.x = float(self.copy_cloud_key_poses_3d[key_cur]["x"])
            p.y = float(self.copy_cloud_key_poses_3d[key_cur]["y"])
            p.z = float(self.copy_cloud_key_poses_3d[key_cur]["z"])
            marker_node.points.append(p)
            marker_edge.points.append(p)
            p.x = float(self.copy_cloud_key_poses_3d[key_pre]["x"])
            p.y = float(self.copy_cloud_key_poses_3d[key_pre]["y"])
            p.z = float(self.copy_cloud_key_poses_3d[key_pre]["z"])
            marker_node.points.append(p)
            marker_edge.points.append(p)

        marker_array.markers.append(marker_node)
        marker_array.markers.append(marker_edge)
        self.pub_loop_constraint_edge.publish(marker_array)

    def update_initial_guess(self) -> None:
        # save current transformation before any processing
        self.incremental_odometry_affine_front = MapOptimization.trans_to_mat(
            self.transform_tobe_mapped
        )

        # initialization
        if self.cloud_key_poses_3d.shape[0] == 0:
            self.transform_tobe_mapped[0] = self.cloud_info.imu_roll_init
            self.transform_tobe_mapped[1] = self.cloud_info.imu_pitch_init
            self.transform_tobe_mapped[2] = self.cloud_info.imu_yaw_init

            if not self.use_imu_heading_initialization:
                self.transform_tobe_mapped[2] = 0.0

            self.last_imu_transformation = matrix_from_rpy_xyz(
                self.cloud_info.imu_roll_init,
                self.cloud_info.imu_pitch_init,
                self.cloud_info.imu_yaw_init,
                0.0,
                0.0,
                0.0,
            )  # save imu before return
            return

        # use imu preintegration estimation for pose guess
        if self.cloud_info.odom_available:
            trans_back = matrix_from_rpy_xyz(
                self.cloud_info.initial_guess_roll,
                self.cloud_info.initial_guess_pitch,
                self.cloud_info.initial_guess_yaw,
                self.cloud_info.initial_guess_x,
                self.cloud_info.initial_guess_y,
                self.cloud_info.initial_guess_z,
            )
            if not self.last_imu_pre_trans_available:
                self.last_imu_pre_transformation = trans_back
                self.last_imu_pre_trans_available = True
            else:
                trans_incre = (
                    np.linalg.inv(self.last_imu_pre_transformation) @ trans_back
                )
                trans_tobe = MapOptimization.trans_to_mat(self.transform_tobe_mapped)
                trans_final = trans_tobe @ trans_incre
                x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(trans_final)
                self.transform_tobe_mapped[:] = [roll, pitch, yaw, x, y, z]
                self.last_imu_pre_transformation = trans_back

                self.last_imu_transformation = matrix_from_rpy_xyz(
                    self.cloud_info.imu_roll_init,
                    self.cloud_info.imu_pitch_init,
                    self.cloud_info.imu_yaw_init,
                    0.0,
                    0.0,
                    0.0,
                )  # save imu before return
                return

        # use imu incremental estimation for pose guess (only rotation)
        if self.cloud_info.imu_available:
            trans_back = matrix_from_rpy_xyz(
                self.cloud_info.imu_roll_init,
                self.cloud_info.imu_pitch_init,
                self.cloud_info.imu_yaw_init,
                0.0,
                0.0,
                0.0,
            )
            trans_incre = np.linalg.inv(self.last_imu_transformation) @ trans_back

            trans_tobe = MapOptimization.trans_to_mat(self.transform_tobe_mapped)
            trans_final = trans_tobe @ trans_incre
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(trans_final)
            self.transform_tobe_mapped[:] = [roll, pitch, yaw, x, y, z]

            self.last_imu_transformation = trans_back  # save imu before return

    def extract_for_loop_closure(self) -> None:
        cloud_to_extract = np.empty(0, dtype=KEY_POSE_3D_DTYPE)
        poses = []
        num_poses = self.cloud_key_poses_3d.shape[0]
        for i in range(num_poses - 1, -1, -1):
            if len(poses) <= self.surrounding_keyframe_size:
                poses.append(self.cloud_key_poses_3d[i])
            else:
                break

        if poses:
            cloud_to_extract = np.array(poses, dtype=KEY_POSE_3D_DTYPE)

        self.extract_cloud(cloud_to_extract)

    def extract_nearby(self) -> None:
        # extract all the nearby key poses and downsample them
        cloud_key_poses_3d_xyz = xyz_array(self.cloud_key_poses_3d)
        kdtree_surrounding_key_poses = cKDTree(cloud_key_poses_3d_xyz)  # create kd-tree
        point_search_ind = kdtree_surrounding_key_poses.query_ball_point(
            cloud_key_poses_3d_xyz[-1], self.surrounding_keyframe_search_radius
        )
        surrounding_key_poses = self.cloud_key_poses_3d[
            np.asarray(point_search_ind, dtype=np.int64)
        ].copy()

        surrounding_key_poses_ds = voxel_downsample(
            surrounding_key_poses, self.surrounding_keyframe_density
        )
        if surrounding_key_poses_ds.size:
            surrounding_key_poses_ds_xyz = xyz_array(surrounding_key_poses_ds)
            _, nn = kdtree_surrounding_key_poses.query(
                surrounding_key_poses_ds_xyz, k=1
            )
            surrounding_key_poses_ds["intensity"] = self.cloud_key_poses_3d[
                "intensity"
            ][nn]

        # also extract some latest key frames in case the robot rotates in one position
        latest_key_frames = []
        num_poses = self.cloud_key_poses_3d.shape[0]
        for i in range(num_poses - 1, -1, -1):
            if self.time_laser_info_cur - self.cloud_key_poses_6d[i]["time"] < 10.0:
                latest_key_frames.append(self.cloud_key_poses_3d[i])
            else:
                break
        if latest_key_frames:
            surrounding_key_poses_ds = np.concatenate(
                (
                    surrounding_key_poses_ds,
                    np.array(latest_key_frames, dtype=KEY_POSE_3D_DTYPE),
                )
            )

        self.extract_cloud(surrounding_key_poses_ds)

    def extract_cloud(self, cloud_to_extract: np.ndarray) -> None:
        # fuse the map
        self.laser_cloud_corner_from_map = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_from_map = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_corner_from_map_ds = np.empty(0, dtype=POINT_DTYPE)
        self.laser_cloud_surf_from_map_ds = np.empty(0, dtype=POINT_DTYPE)
        laser_cloud_corner = []
        laser_cloud_surf = []
        for i in range(cloud_to_extract.shape[0]):
            if (
                points_distance(cloud_to_extract[i], self.cloud_key_poses_3d[-1])
                > self.surrounding_keyframe_search_radius
            ):
                continue

            this_key_ind = int(round(cloud_to_extract[i]["intensity"]))
            if this_key_ind in self.laser_cloud_map_container:
                # transform cloud available
                laser_cloud_corner_temp, laser_cloud_surf_temp = (
                    self.laser_cloud_map_container[this_key_ind]
                )
            else:
                # transform cloud not available
                laser_cloud_corner_temp = self.transform_point_cloud(
                    self.corner_cloud_key_frames[this_key_ind],
                    self.cloud_key_poses_6d[this_key_ind],
                )
                laser_cloud_surf_temp = self.transform_point_cloud(
                    self.surf_cloud_key_frames[this_key_ind],
                    self.cloud_key_poses_6d[this_key_ind],
                )
                self.laser_cloud_map_container[this_key_ind] = (
                    laser_cloud_corner_temp,
                    laser_cloud_surf_temp,
                )

            laser_cloud_corner.append(laser_cloud_corner_temp)
            laser_cloud_surf.append(laser_cloud_surf_temp)

        # Downsample the surrounding corner key frames (or map)
        if laser_cloud_corner:
            self.laser_cloud_corner_from_map = np.concatenate(laser_cloud_corner)
            self.laser_cloud_corner_from_map_ds = voxel_downsample(
                self.laser_cloud_corner_from_map, self.mapping_corner_leaf_size
            )
        # Downsample the surrounding surf key frames (or map)
        if laser_cloud_surf:
            self.laser_cloud_surf_from_map = np.concatenate(laser_cloud_surf)
            self.laser_cloud_surf_from_map_ds = voxel_downsample(
                self.laser_cloud_surf_from_map, self.mapping_surf_leaf_size
            )

        # clear map cache if too large
        if len(self.laser_cloud_map_container) > 1000:
            self.laser_cloud_map_container.clear()

    def extract_surrounding_key_frames(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return

        # if self.lood_closure_enable_flag:
        #     self.extract_for_loop_closure()
        # else:
        #     self.extract_nearby()

        self.extract_nearby()

    def downsample_current_scan(self) -> None:
        # Downsample cloud from current scan
        self.laser_cloud_corner_last_ds = voxel_downsample(
            self.laser_cloud_corner_last, self.mapping_corner_leaf_size
        )
        self.laser_cloud_corner_last_ds_num = self.laser_cloud_corner_last_ds.shape[0]

        self.laser_cloud_surf_last_ds = voxel_downsample(
            self.laser_cloud_surf_last, self.mapping_surf_leaf_size
        )
        self.laser_cloud_surf_last_ds_num = self.laser_cloud_surf_last_ds.shape[0]

    def update_point_associate_to_map(self) -> None:
        self.trans_point_associate_to_map = MapOptimization.trans_to_mat(
            self.transform_tobe_mapped
        )

    def corner_optimization(self) -> None:
        self.update_point_associate_to_map()

        if self.laser_cloud_corner_last_ds_num == 0:
            return

        point_ori_xyz = xyz_array(self.laser_cloud_corner_last_ds)
        point_sel_xyz = (
            point_ori_xyz @ self.trans_point_associate_to_map[:3, :3].T
            + self.trans_point_associate_to_map[:3, 3]
        )
        point_search_dis, point_search_ind = self.kdtree_corner_from_map.query(
            point_sel_xyz, k=5
        )
        point_search_sq_dis = point_search_dis**2

        valid = np.isfinite(point_search_sq_dis[:, 4])
        valid &= point_search_sq_dis[:, 4] < 1.0
        if not np.any(valid):
            return

        valid_ids = np.flatnonzero(valid)
        nn_ids = point_search_ind[valid]

        map_xyz = xyz_array(self.laser_cloud_corner_from_map_ds)
        nn_xyz = map_xyz[nn_ids]
        c_xyz = np.mean(nn_xyz, axis=1)

        diff = nn_xyz - c_xyz[:, None, :]
        matA1 = np.einsum("nki,nkj->nij", diff, diff) / 5.0

        matD1, matV1 = np.linalg.eigh(matA1)

        line_valid = matD1[:, 2] > 3 * matD1[:, 1]
        if not np.any(line_valid):
            return

        valid_ids = valid_ids[line_valid]
        c_xyz = c_xyz[line_valid]
        point_sel_xyz = point_sel_xyz[valid_ids]

        line_dir = matV1[line_valid, :, 2]
        p1 = c_xyz + 0.1 * line_dir
        p2 = c_xyz - 0.1 * line_dir
        x0 = point_sel_xyz[:, 0]
        y0 = point_sel_xyz[:, 1]
        z0 = point_sel_xyz[:, 2]
        x1 = p1[:, 0]
        y1 = p1[:, 1]
        z1 = p1[:, 2]
        x2 = p2[:, 0]
        y2 = p2[:, 1]
        z2 = p2[:, 2]

        a = (x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)
        b = (x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)
        c = (y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)
        a012 = np.sqrt(a * a + b * b + c * c)

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        l12 = np.sqrt(dx * dx + dy * dy + dz * dz)

        denom = a012 * l12
        nonzero_denom = denom > 1e-12
        if not np.any(nonzero_denom):
            return

        denom = denom[nonzero_denom]
        valid_ids = valid_ids[nonzero_denom]

        a = a[nonzero_denom]
        b = b[nonzero_denom]
        c = c[nonzero_denom]
        a012 = a012[nonzero_denom]

        dx = dx[nonzero_denom]
        dy = dy[nonzero_denom]
        dz = dz[nonzero_denom]
        l12 = l12[nonzero_denom]

        la = (dy * a + dz * b) / denom
        lb = -(dx * a - dz * c) / denom
        lc = -(dx * b + dy * c) / denom

        ld2 = a012 / l12

        s = 1 - 0.9 * np.abs(ld2)
        coeff_valid = s > 0.1
        if not np.any(coeff_valid):
            return

        valid_ids = valid_ids[coeff_valid]
        s = s[coeff_valid]
        la = la[coeff_valid]
        lb = lb[coeff_valid]
        lc = lc[coeff_valid]
        ld2 = ld2[coeff_valid]

        self.laser_cloud_ori_corner_vec[valid_ids] = self.laser_cloud_corner_last_ds[
            valid_ids
        ]
        self.coeff_sel_corner_vec["x"][valid_ids] = s * la
        self.coeff_sel_corner_vec["y"][valid_ids] = s * lb
        self.coeff_sel_corner_vec["z"][valid_ids] = s * lc
        self.coeff_sel_corner_vec["intensity"][valid_ids] = s * ld2
        self.laser_cloud_ori_corner_flag[valid_ids] = True

    def surf_optimization(self) -> None:
        self.update_point_associate_to_map()

        if self.laser_cloud_surf_last_ds_num == 0:
            return

        point_ori_xyz = xyz_array(self.laser_cloud_surf_last_ds)
        point_sel_xyz = (
            point_ori_xyz @ self.trans_point_associate_to_map[:3, :3].T
            + self.trans_point_associate_to_map[:3, 3]
        )
        point_search_dis, point_search_ind = self.kdtree_surf_from_map.query(
            point_sel_xyz, k=5
        )
        point_search_sq_dis = point_search_dis**2

        valid = np.isfinite(point_search_sq_dis[:, 4])
        valid &= point_search_sq_dis[:, 4] < 1.0
        if not np.any(valid):
            return

        valid_ids = np.flatnonzero(valid)
        nn_ids = point_search_ind[valid]

        nn_xyz = xyz_array(self.laser_cloud_surf_from_map_ds)[nn_ids]

        point_sel_valid = point_sel_xyz[valid_ids]
        point_ori_valid = point_ori_xyz[valid_ids]

        mat_b = np.full((nn_xyz.shape[0], 5), -1.0, dtype=np.float64)
        ata = np.einsum("nki,nkj->nij", nn_xyz, nn_xyz)
        atb = np.einsum("nki,nk->ni", nn_xyz, mat_b)

        try:
            mat_x = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            mat_x = np.full((nn_xyz.shape[0], 3), np.nan, dtype=np.float64)

        pa = mat_x[:, 0]
        pb = mat_x[:, 1]
        pc = mat_x[:, 2]
        pd = np.ones_like(pa)

        ps = np.sqrt(pa * pa + pb * pb + pc * pc)
        valid_ps = ps >= 1e-12
        if not np.any(valid_ps):
            return

        pa = pa / np.where(valid_ps, ps, 1.0)
        pb = pb / np.where(valid_ps, ps, 1.0)
        pc = pc / np.where(valid_ps, ps, 1.0)
        pd = pd / np.where(valid_ps, ps, 1.0)

        plane_error = (
            pa[:, None] * nn_xyz[:, :, 0]
            + pb[:, None] * nn_xyz[:, :, 1]
            + pc[:, None] * nn_xyz[:, :, 2]
            + pd[:, None]
        )
        valid_plane = np.all(
            np.abs(plane_error) <= 0.2,
            axis=1,
        )

        pd2 = (
            pa * point_sel_valid[:, 0]
            + pb * point_sel_valid[:, 1]
            + pc * point_sel_valid[:, 2]
            + pd
        )

        point_ori_norm = np.linalg.norm(
            point_ori_valid,
            axis=1,
        )
        valid_norm = point_ori_norm >= 1e-12
        if not np.any(valid_norm):
            return

        s = np.zeros_like(pd2)
        s[valid_norm] = 1.0 - 0.9 * np.abs(pd2[valid_norm]) / np.sqrt(
            point_ori_norm[valid_norm]
        )
        valid_s = s > 0.1

        final_valid = valid_ps & valid_plane & valid_norm & valid_s
        if not np.any(final_valid):
            return

        final_ids = valid_ids[final_valid]
        final_s = s[final_valid]
        final_pa = pa[final_valid]
        final_pb = pb[final_valid]
        final_pc = pc[final_valid]
        final_pd2 = pd2[final_valid]

        self.laser_cloud_ori_surf_vec[final_ids] = self.laser_cloud_surf_last_ds[
            final_ids
        ]
        self.coeff_sel_surf_vec["x"][final_ids] = final_s * final_pa
        self.coeff_sel_surf_vec["y"][final_ids] = final_s * final_pb
        self.coeff_sel_surf_vec["z"][final_ids] = final_s * final_pc
        self.coeff_sel_surf_vec["intensity"][final_ids] = final_s * final_pd2
        self.laser_cloud_ori_surf_flag[final_ids] = True

    def combine_optimization(self) -> None:
        ori_parts = []
        coeff_parts = []

        # combine corner coeffs
        ids = np.flatnonzero(
            self.laser_cloud_ori_corner_flag[: self.laser_cloud_corner_last_ds_num]
        )
        if ids.size:
            ori_parts.append(self.laser_cloud_ori_corner_vec[ids])
            coeff_parts.append(self.coeff_sel_corner_vec[ids])
        # combine surf coeffs
        ids = np.flatnonzero(
            self.laser_cloud_ori_surf_flag[: self.laser_cloud_surf_last_ds_num]
        )
        if ids.size:
            ori_parts.append(self.laser_cloud_ori_surf_vec[ids])
            coeff_parts.append(self.coeff_sel_surf_vec[ids])
        if ori_parts:
            self.laser_cloud_ori = np.concatenate(ori_parts)
            self.coeff_sel = np.concatenate(coeff_parts)
        else:
            self.laser_cloud_ori = np.empty(0, dtype=POINT_DTYPE)
            self.coeff_sel = np.empty(0, dtype=POINT_DTYPE)
        # reset flag for next iteration
        self.laser_cloud_ori_corner_flag[: self.laser_cloud_corner_last_ds_num] = False
        self.laser_cloud_ori_surf_flag[: self.laser_cloud_surf_last_ds_num] = False

    def lm_optimization(self, iter_count: int) -> bool:
        # This optimization is from the original loam_velodyne by Ji Zhang,
        # need to cope with coordinate transformation
        # lidar <- camera      ---     camera <- lidar
        # x = z                ---     x = y
        # y = x                ---     y = z
        # z = y                ---     z = x
        # roll = yaw           ---     roll = pitch
        # pitch = roll         ---     pitch = yaw
        # yaw = pitch          ---     yaw = roll

        # lidar -> camera
        srx = np.sin(self.transform_tobe_mapped[1])
        crx = np.cos(self.transform_tobe_mapped[1])
        sry = np.sin(self.transform_tobe_mapped[2])
        cry = np.cos(self.transform_tobe_mapped[2])
        srz = np.sin(self.transform_tobe_mapped[0])
        crz = np.cos(self.transform_tobe_mapped[0])

        laser_cloud_sel_num = self.laser_cloud_ori.shape[0]
        if laser_cloud_sel_num < 50:
            return False

        # lidar -> camera
        point_ori_x = self.laser_cloud_ori["y"].astype(np.float64, copy=False)
        point_ori_y = self.laser_cloud_ori["z"].astype(np.float64, copy=False)
        point_ori_z = self.laser_cloud_ori["x"].astype(np.float64, copy=False)
        # lidar -> camera
        coeff_x = self.coeff_sel["y"].astype(np.float64, copy=False)
        coeff_y = self.coeff_sel["z"].astype(np.float64, copy=False)
        coeff_z = self.coeff_sel["x"].astype(np.float64, copy=False)
        coeff_intensity = self.coeff_sel["intensity"].astype(np.float64, copy=False)
        # in camera
        arx = (
            (
                crx * sry * srz * point_ori_x
                + crx * crz * sry * point_ori_y
                - srx * sry * point_ori_z
            )
            * coeff_x
            + (-srx * srz * point_ori_x - crz * srx * point_ori_y - crx * point_ori_z)
            * coeff_y
            + (
                crx * cry * srz * point_ori_x
                + crx * cry * crz * point_ori_y
                - cry * srx * point_ori_z
            )
            * coeff_z
        )
        ary = (
            (cry * srx * srz - crz * sry) * point_ori_x
            + (sry * srz + cry * crz * srx) * point_ori_y
            + crx * cry * point_ori_z
        ) * coeff_x + (
            (-cry * crz - srx * sry * srz) * point_ori_x
            + (cry * srz - crz * srx * sry) * point_ori_y
            - crx * sry * point_ori_z
        ) * coeff_z
        arz = (
            (
                (crz * srx * sry - cry * srz) * point_ori_x
                + (-cry * crz - srx * sry * srz) * point_ori_y
            )
            * coeff_x
            + (crx * crz * point_ori_x - crx * srz * point_ori_y) * coeff_y
            + (
                (sry * srz + cry * crz * srx) * point_ori_x
                + (crz * sry - cry * srx * srz) * point_ori_y
            )
            * coeff_z
        )
        # lidar -> camera
        matA = np.column_stack((arz, arx, ary, coeff_z, coeff_x, coeff_y))
        matB = -coeff_intensity

        matAtA = matA.T @ matA
        matAtB = matA.T @ matB
        matX, _, _, _ = np.linalg.lstsq(matAtA, matAtB, rcond=None)
        matX = np.asarray(matX, dtype=np.float64).reshape(6)

        if iter_count == 0:
            matE, matV = np.linalg.eigh(matAtA)
            matE_threshold = np.full(6, 100, dtype=np.float64)

            keep = matE >= matE_threshold
            self.is_degenerate = not np.all(keep)
            self.mat_P = matV @ np.diag(keep.astype(np.float64)) @ matV.T

        if self.is_degenerate:
            matX = self.mat_P @ matX

        self.transform_tobe_mapped += matX

        delta_r = np.linalg.norm(np.rad2deg(matX[:3]))
        delta_t = np.linalg.norm(matX[3:] * 100.0)

        if delta_r < 0.05 and delta_t < 0.05:
            return True  # converged
        return False  # kep optimizing

    def scan_to_map_optimization(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return

        if (
            self.laser_cloud_corner_last_ds_num > self.edge_feature_min_valid_num
            and self.laser_cloud_surf_last_ds_num > self.surf_feature_min_valid_num
        ):
            self.kdtree_corner_from_map = cKDTree(
                xyz_array(self.laser_cloud_corner_from_map_ds)
            )
            self.kdtree_surf_from_map = cKDTree(
                xyz_array(self.laser_cloud_surf_from_map_ds)
            )

            for iter_count in range(0, 30):
                self.laser_cloud_ori = np.empty(0, dtype=POINT_DTYPE)
                self.coeff_sel = np.empty(0, dtype=POINT_DTYPE)

                self.corner_optimization()
                self.surf_optimization()

                self.combine_optimization()

                if self.lm_optimization(iter_count):
                    break

            self.transform_update()
        else:
            self.get_logger().warning(
                f"Not enough features! Only {self.laser_cloud_corner_last_ds_num} edge "
                f"and {self.laser_cloud_surf_last_ds_num} planar features available."
            )

    def transform_update(self) -> None:
        if self.cloud_info.imu_available:
            if abs(self.cloud_info.imu_pitch_init) < 1.4:
                imu_weight = float(self.imu_rpy_weight)

                # slerp roll
                self.transform_tobe_mapped[0] = slerp_single_axis(
                    self.transform_tobe_mapped[0],
                    self.cloud_info.imu_roll_init,
                    "roll",
                    imu_weight,
                )

                # slerp pitch
                self.transform_tobe_mapped[1] = slerp_single_axis(
                    self.transform_tobe_mapped[1],
                    self.cloud_info.imu_pitch_init,
                    "pitch",
                    imu_weight,
                )

        self.transform_tobe_mapped[0] = np.clip(
            self.transform_tobe_mapped[0],
            -self.rotation_tollerance,
            self.rotation_tollerance,
        )
        self.transform_tobe_mapped[1] = np.clip(
            self.transform_tobe_mapped[1],
            -self.rotation_tollerance,
            self.rotation_tollerance,
        )
        self.transform_tobe_mapped[5] = np.clip(
            self.transform_tobe_mapped[5], -self.z_tollerance, self.z_tollerance
        )

        self.incremental_odometry_affine_back = MapOptimization.trans_to_mat(
            self.transform_tobe_mapped
        )

    def save_frame(self) -> bool:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return True

        if self.sensor == SensorType.LIVOX:
            if self.time_laser_info_cur - self.cloud_key_poses_6d[-1]["time"] > 1.0:
                return True

        trans_start = MapOptimization.point_to_mat(self.cloud_key_poses_6d[-1])
        trans_final = MapOptimization.trans_to_mat(self.transform_tobe_mapped)
        trans_between = np.linalg.inv(trans_start) @ trans_final
        x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(trans_between)

        if (
            abs(roll) < self.surroundingkeyframe_adding_angle_threshold
            and abs(pitch) < self.surroundingkeyframe_adding_angle_threshold
            and abs(yaw) < self.surroundingkeyframe_adding_angle_threshold
            and np.linalg.norm([x, y, z])
            < self.surroundingkeyframe_adding_dist_threshold
        ):
            return False

        return True

    def add_odom_factor(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            prior_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-2, 1e-2, np.pi**2, 1e8, 1e8, 1e8], dtype=np.float64)
            )  # rad*rad, meter*meter
            prior_pose = MapOptimization.trans_to_pose3(self.transform_tobe_mapped)
            self.gtsam_graph.add(gtsam.PriorFactorPose3(0, prior_pose, prior_noise))
            self.initial_estimate.insert(0, prior_pose)
        else:
            odometry_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4], dtype=np.float64)
            )
            pose_from = MapOptimization.point_to_pose3(self.cloud_key_poses_6d[-1])
            pose_to = MapOptimization.trans_to_pose3(self.transform_tobe_mapped)
            i = self.cloud_key_poses_3d.shape[0]
            self.gtsam_graph.add(
                gtsam.BetweenFactorPose3(
                    i - 1, i, pose_from.between(pose_to), odometry_noise
                )
            )
            self.initial_estimate.insert(i, pose_to)

    def add_gps_factor(self) -> None:
        if not self.gps_queue:
            return

        # wait for system initialized and settles down
        if self.cloud_key_poses_3d.shape[0] == 0:
            return
        else:
            if (
                points_distance(self.cloud_key_poses_3d[0], self.cloud_key_poses_3d[-1])
                < 5.0
            ):
                return

        # pose covariance small, no need to correct
        if (
            self.pose_covariance[3, 3] < self.pose_cov_threshold
            and self.pose_covariance[4, 4] < self.pose_cov_threshold
        ):
            return

        while self.gps_queue:
            gps_t = stamp_to_sec(self.gps_queue[0].header.stamp)
            if gps_t < self.time_laser_info_cur - 0.2:
                # message too old
                self.gps_queue.popleft()
            elif gps_t > self.time_laser_info_cur + 0.2:
                # message too new
                break
            else:
                this_gps = self.gps_queue.popleft()

                # GPS too noisy, skip
                noise_x = this_gps.pose.position_covariance[0]
                noise_y = this_gps.pose.position_covariance[7]
                noise_z = this_gps.pose.position_covariance[14]
                if noise_x > self.gps_cov_threshold or noise_y > self.gps_cov_threshold:
                    continue
                gps_x = this_gps.pose.pose.position.x
                gps_y = this_gps.pose.pose.position.y
                gps_z = this_gps.pose.pose.position.z
                if not self.use_gps_elevation:
                    gps_z = self.transform_tobe_mapped[5]
                    noise_z = 0.01

                # GPS not properly initialized (0,0,0)
                if abs(gps_x) < 1e-6 and abs(gps_y) < 1e-6:
                    continue

                # Add GPS every a few meters
                cur_gps_point = np.array([gps_x, gps_y, gps_z], dtype=np.float64)
                if np.linalg.norm(cur_gps_point - self.last_gps_point) < 5.0:
                    continue
                else:
                    self.last_gps_point = cur_gps_point

                gps_noise = gtsam.noiseModel.Diagonal.Variances(
                    np.array(
                        [max(noise_x, 1.0), max(noise_y, 1.0), max(noise_z, 1.0)],
                        dtype=np.float64,
                    )
                )
                self.gtsam_graph.add(
                    gtsam.GPSFactor(
                        self.cloud_key_poses_3d.shape[0],
                        gtsam.Point3(gps_x, gps_y, gps_z),
                        gps_noise,
                    )
                )

                self.a_loop_is_closed = True
                break

    def add_loop_factor(self) -> None:
        if not self.loop_index_queue:
            return

        for (index_from, index_to), pose_between, noise_between in zip(
            self.loop_index_queue, self.loop_pose_queue, self.loop_noise_queue
        ):
            self.gtsam_graph.add(
                gtsam.BetweenFactorPose3(
                    index_from, index_to, pose_between, noise_between
                )
            )

        self.loop_index_queue.clear()
        self.loop_pose_queue.clear()
        self.loop_noise_queue.clear()
        self.a_loop_is_closed = True

    def save_key_frames_and_factor(self) -> None:
        if not self.save_frame():
            return

        # odom factor
        self.add_odom_factor()

        # gps factor
        self.add_gps_factor()

        # loop factor
        self.add_loop_factor()

        # print("****************************************************")
        # self.gtsam_graph.print("GTSAM Graph:\n")

        # update iSAM
        self.isam.update(self.gtsam_graph, self.initial_estimate)
        self.isam.update()

        if self.a_loop_is_closed:
            for _ in range(5):
                self.isam.update()

        self.gtsam_graph.resize(0)
        self.initial_estimate.clear()

        # save key poses
        this_pose_3d = np.zeros(1, dtype=KEY_POSE_3D_DTYPE)
        this_pose_6d = np.zeros(1, dtype=POINT_POSE_DTYPE)

        self.isam_current_estimate = self.isam.calculateEstimate()
        latest_estimate = self.isam_current_estimate.atPose3(
            self.isam_current_estimate.size() - 1
        )
        # print("****************************************************")
        # self.isam_current_estimate.print("Current estimate: ")

        trans = np.asarray(latest_estimate.translation(), dtype=np.float64).reshape(3)
        this_pose_3d[0]["x"] = trans[0]
        this_pose_3d[0]["y"] = trans[1]
        this_pose_3d[0]["z"] = trans[2]
        this_pose_3d[0]["intensity"] = self.cloud_key_poses_3d.shape[
            0
        ]  # this can be used as index
        self.cloud_key_poses_3d = np.concatenate(
            (self.cloud_key_poses_3d, this_pose_3d)
        )

        this_pose_6d[0]["x"] = this_pose_3d[0]["x"]
        this_pose_6d[0]["y"] = this_pose_3d[0]["y"]
        this_pose_6d[0]["z"] = this_pose_3d[0]["z"]
        this_pose_6d[0]["intensity"] = this_pose_3d[0][
            "intensity"
        ]  # this can be used as index
        this_pose_6d[0]["roll"] = latest_estimate.rotation().roll()
        this_pose_6d[0]["pitch"] = latest_estimate.rotation().pitch()
        this_pose_6d[0]["yaw"] = latest_estimate.rotation().yaw()
        this_pose_6d[0]["time"] = self.time_laser_info_cur
        self.cloud_key_poses_6d = np.concatenate(
            (self.cloud_key_poses_6d, this_pose_6d)
        )

        # print("****************************************************")
        # print("Pose covariance:")
        # print(self.isam.marginalCovariance(self.isam_current_estimate.size() - 1))
        self.pose_covariance = self.isam.marginalCovariance(
            self.isam_current_estimate.size() - 1
        )

        # save updated transform
        self.transform_tobe_mapped[0] = latest_estimate.rotation().roll()
        self.transform_tobe_mapped[1] = latest_estimate.rotation().pitch()
        self.transform_tobe_mapped[2] = latest_estimate.rotation().yaw()
        self.transform_tobe_mapped[3] = trans[0]
        self.transform_tobe_mapped[4] = trans[1]
        self.transform_tobe_mapped[5] = trans[2]

        # save all the received edge and surf points
        self.corner_cloud_key_frames.append(self.laser_cloud_corner_last_ds.copy())
        self.surf_cloud_key_frames.append(self.laser_cloud_surf_last_ds.copy())

        # save path for visualization
        self.update_path(this_pose_6d[0])

    def correct_poses(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return

        if self.a_loop_is_closed:
            # clear map cache
            self.laser_cloud_map_container.clear()
            # clear path
            self.global_path.poses.clear()
            # update key poses
            num_poses = self.isam_current_estimate.size()
            for i in range(num_poses):
                pose = self.isam_current_estimate.atPose3(i)
                trans = np.asarray(pose.translation(), dtype=np.float64).reshape(3)
                self.cloud_key_poses_3d[i]["x"] = trans[0]
                self.cloud_key_poses_3d[i]["y"] = trans[1]
                self.cloud_key_poses_3d[i]["z"] = trans[2]

                self.cloud_key_poses_6d[i]["x"] = self.cloud_key_poses_3d[i]["x"]
                self.cloud_key_poses_6d[i]["y"] = self.cloud_key_poses_3d[i]["y"]
                self.cloud_key_poses_6d[i]["z"] = self.cloud_key_poses_3d[i]["z"]
                self.cloud_key_poses_6d[i]["roll"] = pose.rotation().roll()
                self.cloud_key_poses_6d[i]["pitch"] = pose.rotation().pitch()
                self.cloud_key_poses_6d[i]["yaw"] = pose.rotation().yaw()

                self.update_path(self.cloud_key_poses_6d[i])

            self.a_loop_is_closed = False

    def update_path(self, pose_in: np.ndarray) -> None:
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp.sec = int(pose_in["time"])
        pose_stamped.header.stamp.nanosec = int(
            (float(pose_in["time"]) - pose_stamped.header.stamp.sec) * 1e9
        )
        pose_stamped.header.frame_id = self.odometry_frame
        pose_stamped.pose.position.x = float(pose_in["x"])
        pose_stamped.pose.position.y = float(pose_in["y"])
        pose_stamped.pose.position.z = float(pose_in["z"])
        q = quaternion_xyzw_from_rpy(pose_in["roll"], pose_in["pitch"], pose_in["yaw"])
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]

        self.global_path.poses.append(pose_stamped)

    def publish_odometry(self) -> None:
        # Publish odometry for ROS (global)
        laser_odometry_ros = Odometry()
        laser_odometry_ros.header.stamp = self.time_laser_info_stamp
        laser_odometry_ros.header.frame_id = self.odometry_frame
        laser_odometry_ros.child_frame_id = "odom_mapping"
        laser_odometry_ros.pose.pose.position.x = float(self.transform_tobe_mapped[3])
        laser_odometry_ros.pose.pose.position.y = float(self.transform_tobe_mapped[4])
        laser_odometry_ros.pose.pose.position.z = float(self.transform_tobe_mapped[5])
        quat_tf = quaternion_xyzw_from_rpy(
            self.transform_tobe_mapped[0],
            self.transform_tobe_mapped[1],
            self.transform_tobe_mapped[2],
        )
        laser_odometry_ros.pose.pose.orientation.x = quat_tf[0]
        laser_odometry_ros.pose.pose.orientation.y = quat_tf[1]
        laser_odometry_ros.pose.pose.orientation.z = quat_tf[2]
        laser_odometry_ros.pose.pose.orientation.w = quat_tf[3]
        self.pub_laser_odometry_global.publish(laser_odometry_ros)

        # Publish TF
        trans_odom_to_lidar = TransformStamped()
        trans_odom_to_lidar.header.stamp = self.time_laser_info_stamp
        trans_odom_to_lidar.header.frame_id = self.odometry_frame
        trans_odom_to_lidar.child_frame_id = "lidar_link"
        trans_odom_to_lidar.transform.translation.x = float(
            self.transform_tobe_mapped[3]
        )
        trans_odom_to_lidar.transform.translation.y = float(
            self.transform_tobe_mapped[4]
        )
        trans_odom_to_lidar.transform.translation.z = float(
            self.transform_tobe_mapped[5]
        )
        trans_odom_to_lidar.transform.rotation.x = quat_tf[0]
        trans_odom_to_lidar.transform.rotation.y = quat_tf[1]
        trans_odom_to_lidar.transform.rotation.z = quat_tf[2]
        trans_odom_to_lidar.transform.rotation.w = quat_tf[3]
        self.br.sendTransform(trans_odom_to_lidar)

        # Publish odometry for ROS (incremental)
        if not self.last_incre_odom_pub_flag:
            self.last_incre_odom_pub_flag = True
            self.laser_odom_incremental = deepcopy(laser_odometry_ros)
            self.incre_odom_affine = MapOptimization.trans_to_mat(
                self.transform_tobe_mapped
            )
        else:
            affine_incre = (
                np.linalg.inv(self.incremental_odometry_affine_front)
                @ self.incremental_odometry_affine_back
            )
            self.incre_odom_affine = self.incre_odom_affine @ affine_incre
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(self.incre_odom_affine)
            if self.cloud_info.imu_available:
                if abs(self.cloud_info.imu_pitch_init) < 1.4:
                    imu_weight = 0.1

                    # slerp roll
                    roll = slerp_single_axis(
                        roll, self.cloud_info.imu_roll_init, "roll", imu_weight
                    )

                    # slerp pitch
                    pitch = slerp_single_axis(
                        pitch, self.cloud_info.imu_pitch_init, "pitch", imu_weight
                    )
            self.laser_odom_incremental.header.stamp = self.time_laser_info_stamp
            self.laser_odom_incremental.header.frame_id = self.odometry_frame
            self.laser_odom_incremental.child_frame_id = "odom_mapping"
            self.laser_odom_incremental.pose.pose.position.x = x
            self.laser_odom_incremental.pose.pose.position.y = y
            self.laser_odom_incremental.pose.pose.position.z = z
            quat_tf = quaternion_xyzw_from_rpy(roll, pitch, yaw)
            self.laser_odom_incremental.pose.pose.orientation.x = quat_tf[0]
            self.laser_odom_incremental.pose.pose.orientation.y = quat_tf[1]
            self.laser_odom_incremental.pose.pose.orientation.z = quat_tf[2]
            self.laser_odom_incremental.pose.pose.orientation.w = quat_tf[3]
            if self.is_degenerate:
                self.laser_odom_incremental.pose.covariance[0] = 1.0
            else:
                self.laser_odom_incremental.pose.covariance[0] = 0.0
        self.pub_laser_odometry_incremental.publish(self.laser_odom_incremental)

    def publish_frames(self) -> None:
        if self.cloud_key_poses_3d.shape[0] == 0:
            return
        # publish key poses
        publish_cloud(
            self.pub_key_poses,
            self.cloud_key_poses_3d,
            self.time_laser_info_stamp,
            self.odometry_frame,
        )
        # Publish surrounding key frames
        publish_cloud(
            self.pub_recent_key_frames,
            self.laser_cloud_surf_from_map_ds,
            self.time_laser_info_stamp,
            self.odometry_frame,
        )
        # publish registered key frame
        pose = MapOptimization.trans_to_point_pose(self.transform_tobe_mapped)
        if self.pub_recent_key_frame.get_subscription_count() != 0:
            cloud_out = np.concatenate(
                (
                    self.transform_point_cloud(self.laser_cloud_corner_last_ds, pose),
                    self.transform_point_cloud(self.laser_cloud_surf_last_ds, pose),
                )
            )
            publish_cloud(
                self.pub_recent_key_frame,
                cloud_out,
                self.time_laser_info_stamp,
                self.odometry_frame,
            )
        # publish registered high-res raw cloud
        if self.pub_cloud_registered_raw.get_subscription_count() != 0:
            cloud_out = pointcloud2_to_numpy(self.cloud_info.cloud_deskewed)
            cloud_out = self.transform_point_cloud(cloud_out, pose)
            publish_cloud(
                self.pub_cloud_registered_raw,
                cloud_out,
                self.time_laser_info_stamp,
                self.odometry_frame,
            )
        # publish path
        if self.pub_path.get_subscription_count() != 0:
            self.global_path.header.stamp = self.time_laser_info_stamp
            self.global_path.header.frame_id = self.odometry_frame
            self.pub_path.publish(self.global_path)

    def save_map_to_directory(self, directory: str, resolution: float) -> bool:
        print("****************************************************")
        print("Saving map to pcd files ...")
        try:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)

            self.write_pcd_ascii(
                os.path.join(directory, "trajectory.pcd"), self.cloud_key_poses_3d
            )
            self.write_pcd_ascii(
                os.path.join(directory, "transformations.pcd"), self.cloud_key_poses_6d
            )

            corners, surfs = [], []
            for i in range(self.cloud_key_poses_3d.size):
                corners.append(
                    self.transform_point_cloud(
                        self.corner_cloud_key_frames[i], self.cloud_key_poses_6d[i]
                    )
                )
                surfs.append(
                    self.transform_point_cloud(
                        self.surf_cloud_key_frames[i], self.cloud_key_poses_6d[i]
                    )
                )
            global_corner_cloud = (
                np.concatenate(corners) if corners else np.empty(0, dtype=POINT_DTYPE)
            )
            global_surf_cloud = (
                np.concatenate(surfs) if surfs else np.empty(0, dtype=POINT_DTYPE)
            )

            global_corner_cloud_ds = (
                voxel_downsample(global_corner_cloud, resolution)
                if resolution
                else global_corner_cloud
            )
            global_surf_cloud_ds = (
                voxel_downsample(global_surf_cloud, resolution)
                if resolution
                else global_surf_cloud
            )

            self.write_pcd_ascii(
                os.path.join(directory, "CornerMap.pcd"), global_corner_cloud_ds
            )
            self.write_pcd_ascii(
                os.path.join(directory, "SurfMap.pcd"), global_surf_cloud_ds
            )
            self.write_pcd_ascii(
                os.path.join(directory, "GlobalMap.pcd"),
                np.concatenate((global_corner_cloud, global_surf_cloud)),
            )

            print("****************************************************")
            print("Saving map to pcd files completed")
            return True
        except Exception as exc:
            self.get_logger().error(f"Failed to save map: {exc}")
            return False

    def write_pcd_ascii(self, filename: str, cloud: np.ndarray) -> None:
        if cloud.dtype.names is None:
            raise ValueError("cloud must be a structured NumPy array")

        fields = cloud.dtype.names
        num_points = cloud.shape[0]

        sizes = []
        types = []
        output_dtypes = {}

        for name in fields:
            if name == "time":
                dtype = np.dtype(np.float64)
            elif name in ("x", "y", "z", "intensity", "roll", "pitch", "yaw"):
                dtype = np.dtype(np.float32)
            else:
                dtype = cloud.dtype.fields[name][0]

            output_dtypes[name] = dtype

            if np.issubdtype(dtype, np.floating):
                pcd_type = "F"
            elif np.issubdtype(dtype, np.signedinteger):
                pcd_type = "I"
            elif np.issubdtype(dtype, np.unsignedinteger):
                pcd_type = "U"
            else:
                raise TypeError(f"Unsupported dtype for PCD field '{name}': {dtype}")

            sizes.append(str(dtype.itemsize))
            types.append(pcd_type)

        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            f"FIELDS {' '.join(fields)}\n"
            f"SIZE {' '.join(sizes)}\n"
            f"TYPE {' '.join(types)}\n"
            f"COUNT {' '.join(['1'] * len(fields))}\n"
            f"WIDTH {num_points}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {num_points}\n"
            "DATA ascii\n"
        )

        with open(filename, "w", encoding="ascii") as f:
            f.write(header)

            for point in cloud:
                values = []

                for name in fields:
                    dtype = output_dtypes[name]
                    value = point[name]

                    if np.issubdtype(dtype, np.floating):
                        value = dtype.type(value)
                        values.append(f"{float(value):.9g}")
                    else:
                        values.append(str(int(value)))

                f.write(" ".join(values) + "\n")


def main(args=None):
    rclpy.init(args=args)

    exec = rclpy.executors.SingleThreadedExecutor()

    mo = MapOptimization()
    exec.add_node(mo)

    mo.get_logger().info("\033[1;32m----> Map Optimization Started.\033[0m")

    loop_thread = threading.Thread(target=mo.loop_closure_thread, daemon=True)
    visualize_map_thread = threading.Thread(
        target=mo.visualize_global_map_thread, daemon=True
    )

    loop_thread.start()
    visualize_map_thread.start()

    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        mo.destroy_node()
        rclpy.shutdown()
        loop_thread.join(timeout=1.0)
        visualize_map_thread.join(timeout=1.0)
