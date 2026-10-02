# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2020 Tixiao Shan
# Copyright 2021 Christoph Gruber
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
# Original File: src/imageProjection.cpp
# Original Author: Tixiao Shan
# Original Source: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
#
# Modifications: This file is a Python port of the original C++ implementation
#                by Eugene Auh in 2026.
# -----------------------------------------------------------------------------
from collections import deque
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Header

from lio_sam_py.utility import (
    ParamServer,
    SensorType,
    POINT_FIELDS,
    imu_angular_to_ros_angular,
    imu_rpy_to_ros_rpy,
    point_distance,
    pointcloud2_to_numpy,
    publish_cloud,
    stamp_to_sec,
    qos,
    qos_imu,
    qos_lidar,
)
from lio_sam_msgs.msg import CloudInfo

velodyne_point_xyzirt = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("ring", np.uint16),
        ("time", np.float32),
    ]
)

ouster_point_xyzirt = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("t", np.uint32),
        ("reflectivity", np.uint16),
        ("ring", np.uint8),
        ("noise", np.uint16),
        ("range", np.uint32),
    ]
)

# only tested with Livox Mid-360 LiDAR.
livox_point_xyzirt = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("tag", np.uint8),
        ("line", np.uint8),
        ("timestamp", np.float64),
    ]
)

# Use the Velodyne point format as a common representation
point_xyzirt = velodyne_point_xyzirt

QUEUE_LENGTH = 2000


class ImageProjection(ParamServer):
    def __init__(self):
        super().__init__(node_name="image_projection")

        self.imu_lock = threading.Lock()
        self.odo_lock = threading.Lock()

        self.callback_group_lidar = MutuallyExclusiveCallbackGroup()
        self.sub_laser_cloud = self.create_subscription(
            PointCloud2,
            self.point_cloud_topic,
            self.cloud_handler,
            qos_lidar,
            callback_group=self.callback_group_lidar,
        )

        self.pub_extracted_cloud = self.create_publisher(
            PointCloud2, "lio_sam/deskew/cloud_deskewed", qos
        )
        self.pub_laser_cloud_info = self.create_publisher(
            CloudInfo, "lio_sam/deskew/cloud_info", qos
        )

        self.callback_group_imu = MutuallyExclusiveCallbackGroup()
        self.sub_imu = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_handler,
            qos_imu,
            callback_group=self.callback_group_imu,
        )
        self.imu_queue = deque()

        self.callback_group_odom = MutuallyExclusiveCallbackGroup()
        self.sub_odom = self.create_subscription(
            Odometry,
            f"{self.odom_topic}_incremental",
            self.odom_handler,
            qos_imu,
            callback_group=self.callback_group_odom,
        )
        self.odom_queue = deque()

        self.cloud_queue = deque()
        self.current_cloud_msg = PointCloud2()

        self.imu_time = np.zeros(QUEUE_LENGTH, dtype=np.float64)
        self.imu_rot_x = np.zeros(QUEUE_LENGTH, dtype=np.float64)
        self.imu_rot_y = np.zeros(QUEUE_LENGTH, dtype=np.float64)
        self.imu_rot_z = np.zeros(QUEUE_LENGTH, dtype=np.float64)

        self.imu_pointer_cur = 0
        self.first_point_flag = True
        self.trans_start_inverse = np.eye(4, dtype=np.float64)
        self.trans_start_rot = np.eye(3, dtype=np.float64)

        self.laser_cloud_in = np.empty(0, dtype=point_xyzirt)
        self.tmp_ouster_cloud_in = np.empty(0, dtype=ouster_point_xyzirt)
        self.tmp_livox_cloud_in = np.empty(0, dtype=livox_point_xyzirt)
        self.full_cloud = np.empty(0, dtype=point_xyzirt)
        self.extracted_cloud = np.empty(0, dtype=point_xyzirt)

        self.ring_flag = 0
        self.deskew_flag = 0
        self.range_mat = np.full(
            (self.n_scan, self.horizon_scan), np.finfo(np.float32).max, dtype=np.float32
        )

        self.odom_deskew_flag = False
        self.odom_incre_x = 0.0
        self.odom_incre_y = 0.0
        self.odom_incre_z = 0.0

        self.cloud_info = CloudInfo()
        self.time_scan_cur: float = None
        self.time_scan_end: float = None
        self.cloud_header = Header()

        self.column_idn_count_vec = np.zeros(self.n_scan, dtype=np.int32)

        self.allocate_memory()
        self.reset_parameters()
        self.imu_debug_count = 0

    def allocate_memory(self) -> None:
        self.full_cloud = np.empty(self.n_scan * self.horizon_scan, dtype=point_xyzirt)

        self.cloud_info.start_ring_index = [0] * self.n_scan
        self.cloud_info.end_ring_index = [0] * self.n_scan

        self.cloud_info.point_col_ind = [0] * (self.n_scan * self.horizon_scan)
        self.cloud_info.point_range = [0.0] * (self.n_scan * self.horizon_scan)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.laser_cloud_in = np.empty(0, dtype=point_xyzirt)
        self.extracted_cloud = np.empty(0, dtype=point_xyzirt)
        # reset range matrix for range image projection
        self.range_mat.fill(np.finfo(np.float32).max)

        self.imu_pointer_cur = 0
        self.first_point_flag = True
        self.odom_deskew_flag = False

        self.imu_time.fill(0.0)
        self.imu_rot_x.fill(0.0)
        self.imu_rot_y.fill(0.0)
        self.imu_rot_z.fill(0.0)
        self.column_idn_count_vec.fill(0)

    def imu_handler(self, imu_msg: Imu) -> None:
        this_imu = self.imu_converter(imu_msg)

        with self.imu_lock:
            self.imu_queue.append(this_imu)

    def odom_handler(self, odometry_msg: Odometry) -> None:
        with self.odo_lock:
            self.odom_queue.append(odometry_msg)

    def cloud_handler(self, laser_cloud_msg: PointCloud2) -> None:
        if not self.cache_point_cloud(laser_cloud_msg):
            return

        if not self.deskew_info():
            return

        self.project_point_cloud()

        self.cloud_extraction()

        self.publish_clouds()

        self.reset_parameters()

    def cache_point_cloud(self, laser_cloud_msg: PointCloud2) -> bool:
        # cache point cloud
        self.cloud_queue.append(laser_cloud_msg)
        if len(self.cloud_queue) <= 2:
            return False

        # convert cloud
        self.current_cloud_msg = self.cloud_queue.popleft()
        if self.sensor == SensorType.VELODYNE or self.sensor == SensorType.LIVOX:
            self.laser_cloud_in = pointcloud2_to_numpy(
                self.current_cloud_msg, point_xyzirt
            )
        elif self.sensor == SensorType.OUSTER:
            self.tmp_ouster_cloud_in = pointcloud2_to_numpy(
                self.current_cloud_msg, ouster_point_xyzirt
            )
            self.laser_cloud_in = np.empty(
                len(self.tmp_ouster_cloud_in), dtype=point_xyzirt
            )
            src = self.tmp_ouster_cloud_in
            dst = self.laser_cloud_in
            dst["x"] = src["x"]
            dst["y"] = src["y"]
            dst["z"] = src["z"]
            dst["intensity"] = src["intensity"]
            dst["ring"] = src["ring"]
            dst["time"] = src["t"] * 1e-9
        elif self.sensor == SensorType.LIVOX360:
            self.tmp_livox_cloud_in = pointcloud2_to_numpy(
                self.current_cloud_msg, livox_point_xyzirt
            )
            self.laser_cloud_in = np.empty(
                len(self.tmp_livox_cloud_in), dtype=point_xyzirt
            )
            src = self.tmp_livox_cloud_in
            dst = self.laser_cloud_in
            dst["x"] = src["x"]
            dst["y"] = src["y"]
            dst["z"] = src["z"]
            dst["intensity"] = src["intensity"]
            dst["ring"] = src["line"]
            cloud_time = stamp_to_sec(self.current_cloud_msg.header.stamp)
            dst["time"] = src["timestamp"] * 1e-9 - cloud_time
        else:
            self.get_logger().error(f"Unknown sensor type: {self.sensor}")
            rclpy.shutdown()

        # get timestamp
        self.cloud_header = self.current_cloud_msg.header
        self.time_scan_cur = stamp_to_sec(self.cloud_header.stamp)
        self.time_scan_end = self.time_scan_cur + float(self.laser_cloud_in["time"][-1])

        # remove Nan
        valid = (
            np.isfinite(self.laser_cloud_in["x"])
            & np.isfinite(self.laser_cloud_in["y"])
            & np.isfinite(self.laser_cloud_in["z"])
        )
        self.laser_cloud_in = self.laser_cloud_in[valid]

        # check ring channel
        # we will skip the ring check in case of velodyne
        # - as we calculate the ring value downstream
        if self.ring_flag == 0:
            self.ring_flag = -1
            field_names = self.laser_cloud_in.dtype.names
            if "ring" in field_names:
                self.ring_flag = 1
            elif self.sensor == SensorType.VELODYNE:
                self.ring_flag = 2
            else:
                self.get_logger().error(
                    "Point cloud ring channel not available, "
                    "please configure your point cloud data!"
                )
                rclpy.shutdown()

        # check point time
        if self.deskew_flag == 0:
            self.deskew_flag = -1
            field_names = self.laser_cloud_in.dtype.names
            if "time" in field_names or "t" in field_names:
                self.deskew_flag = 1
            if self.deskew_flag == -1:
                self.get_logger().warn(
                    "Point cloud timestamp not available, deskew function disabled, "
                    "system will drift significantly!"
                )

        return True

    def deskew_info(self) -> bool:
        with self.imu_lock, self.odo_lock:
            # make sure IMU data available for the scan
            if (
                not self.imu_queue
                or stamp_to_sec(self.imu_queue[0].header.stamp) > self.time_scan_cur
                or stamp_to_sec(self.imu_queue[-1].header.stamp) < self.time_scan_end
            ):
                self.get_logger().info("Waiting for IMU data ...")
                return False

            self.imu_deskew_info()

            self.odom_deskew_info()

            return True

    def imu_deskew_info(self) -> None:
        self.cloud_info.imu_available = False

        while self.imu_queue:
            if stamp_to_sec(self.imu_queue[0].header.stamp) < self.time_scan_cur - 0.01:
                self.imu_queue.popleft()
            else:
                break

        if not self.imu_queue:
            return

        self.imu_pointer_cur = 0

        for this_imu_msg in self.imu_queue:
            current_imu_time = stamp_to_sec(this_imu_msg.header.stamp)

            # get roll, pitch, and yaw estimation for this scan
            if current_imu_time <= self.time_scan_cur:
                (
                    self.cloud_info.imu_roll_init,
                    self.cloud_info.imu_pitch_init,
                    self.cloud_info.imu_yaw_init,
                ) = imu_rpy_to_ros_rpy(this_imu_msg)
            if current_imu_time > self.time_scan_end + 0.01:
                break

            if self.imu_pointer_cur == 0:
                self.imu_rot_x[0] = 0.0
                self.imu_rot_y[0] = 0.0
                self.imu_rot_z[0] = 0.0
                self.imu_time[0] = current_imu_time
                self.imu_pointer_cur += 1
                continue

            # get angular velocity
            angular_x, angular_y, angular_z = imu_angular_to_ros_angular(this_imu_msg)

            # integrate rotation
            time_diff = current_imu_time - self.imu_time[self.imu_pointer_cur - 1]
            self.imu_rot_x[self.imu_pointer_cur] = (
                self.imu_rot_x[self.imu_pointer_cur - 1] + angular_x * time_diff
            )
            self.imu_rot_y[self.imu_pointer_cur] = (
                self.imu_rot_y[self.imu_pointer_cur - 1] + angular_y * time_diff
            )
            self.imu_rot_z[self.imu_pointer_cur] = (
                self.imu_rot_z[self.imu_pointer_cur - 1] + angular_z * time_diff
            )
            self.imu_time[self.imu_pointer_cur] = current_imu_time
            self.imu_pointer_cur += 1

        self.imu_pointer_cur -= 1

        if self.imu_pointer_cur <= 0:
            return

        self.cloud_info.imu_available = True

    def odom_deskew_info(self) -> None:
        self.cloud_info.odom_available = False

        while self.odom_queue:
            if (
                stamp_to_sec(self.odom_queue[0].header.stamp)
                < self.time_scan_cur - 0.01
            ):
                self.odom_queue.popleft()
            else:
                break

        if not self.odom_queue:
            return

        if stamp_to_sec(self.odom_queue[0].header.stamp) > self.time_scan_cur:
            return

        # get start odometry at the beginning of the scan
        start_odom_msg = None

        for this_odom_msg in self.odom_queue:
            start_odom_msg = this_odom_msg

            if stamp_to_sec(start_odom_msg.header.stamp) < self.time_scan_cur:
                continue
            else:
                break

        orientation = R.from_quat(
            [
                start_odom_msg.pose.pose.orientation.x,
                start_odom_msg.pose.pose.orientation.y,
                start_odom_msg.pose.pose.orientation.z,
                start_odom_msg.pose.pose.orientation.w,
            ]
        )
        roll, pitch, yaw = orientation.as_euler("xyz", degrees=False)

        # Initial guess used in map_optimization
        self.cloud_info.initial_guess_x = start_odom_msg.pose.pose.position.x
        self.cloud_info.initial_guess_y = start_odom_msg.pose.pose.position.y
        self.cloud_info.initial_guess_z = start_odom_msg.pose.pose.position.z
        self.cloud_info.initial_guess_roll = roll
        self.cloud_info.initial_guess_pitch = pitch
        self.cloud_info.initial_guess_yaw = yaw

        self.cloud_info.odom_available = True

        # get end odometry at the end of the scan
        self.odom_deskew_flag = False

        if stamp_to_sec(self.odom_queue[-1].header.stamp) < self.time_scan_end:
            return

        end_odom_msg = None

        for this_odom_msg in self.odom_queue:
            end_odom_msg = this_odom_msg

            if stamp_to_sec(end_odom_msg.header.stamp) < self.time_scan_end:
                continue
            else:
                break

        if round(start_odom_msg.pose.covariance[0]) != round(
            end_odom_msg.pose.covariance[0]
        ):
            return

        trans_begin = np.eye(4, dtype=np.float64)

        trans_begin[:3, :3] = R.from_euler(
            "xyz", [roll, pitch, yaw], degrees=False
        ).as_matrix()
        trans_begin[:3, 3] = [
            start_odom_msg.pose.pose.position.x,
            start_odom_msg.pose.pose.position.y,
            start_odom_msg.pose.pose.position.z,
        ]
        trans_end = np.eye(4, dtype=np.float64)
        trans_end[:3, :3] = R.from_quat(
            [
                end_odom_msg.pose.pose.orientation.x,
                end_odom_msg.pose.pose.orientation.y,
                end_odom_msg.pose.pose.orientation.z,
                end_odom_msg.pose.pose.orientation.w,
            ]
        ).as_matrix()
        trans_end[:3, 3] = [
            end_odom_msg.pose.pose.position.x,
            end_odom_msg.pose.pose.position.y,
            end_odom_msg.pose.pose.position.z,
        ]

        trans_bt = np.linalg.inv(trans_begin) @ trans_end

        self.odom_incre_x = np.float32(trans_bt[0, 3])
        self.odom_incre_y = np.float32(trans_bt[1, 3])
        self.odom_incre_z = np.float32(trans_bt[2, 3])

        self.odom_deskew_flag = True

    def find_rotation_batch(
        self, point_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imu_end = self.imu_pointer_cur + 1

        imu_times = np.asanyarray(self.imu_time[:imu_end], dtype=np.float64)

        imu_rot_x = np.asanyarray(self.imu_rot_x[:imu_end], dtype=np.float64)
        imu_rot_y = np.asanyarray(self.imu_rot_y[:imu_end], dtype=np.float64)
        imu_rot_z = np.asanyarray(self.imu_rot_z[:imu_end], dtype=np.float64)

        rot_x_cur = np.interp(point_times, imu_times, imu_rot_x)
        rot_y_cur = np.interp(point_times, imu_times, imu_rot_y)
        rot_z_cur = np.interp(point_times, imu_times, imu_rot_z)

        return rot_x_cur, rot_y_cur, rot_z_cur

    def deskew_points(self, points: np.ndarray, rel_times: np.ndarray) -> np.ndarray:
        if self.deskew_flag == -1 or not self.cloud_info.imu_available:
            return points

        if points.shape[0] == 0:
            return points

        rel_times = np.asarray(rel_times, dtype=np.float64)
        point_times = self.time_scan_cur + rel_times

        rot_x_cur, rot_y_cur, rot_z_cur = self.find_rotation_batch(point_times)

        if self.first_point_flag:
            rot_start = R.from_euler(
                "xyz", [rot_x_cur[0], rot_y_cur[0], rot_z_cur[0]], degrees=False
            ).as_matrix()
            self.trans_start_rot = rot_start
            self.first_point_flag = False

        rots = R.from_euler(
            "xyz", np.column_stack((rot_x_cur, rot_y_cur, rot_z_cur)), degrees=False
        ).as_matrix()
        rel_rots = np.einsum("ij,njk->nik", self.trans_start_rot.T, rots)

        xyz = np.column_stack((points["x"], points["y"], points["z"])).astype(
            np.float64, copy=False
        )
        xyz_deskewed = np.einsum("nij,nj->ni", rel_rots, xyz)

        new_points = points.copy()
        new_points["x"] = xyz_deskewed[:, 0]
        new_points["y"] = xyz_deskewed[:, 1]
        new_points["z"] = xyz_deskewed[:, 2]

        return new_points

    def project_point_cloud(self) -> None:
        cloud_size = len(self.laser_cloud_in)
        if cloud_size == 0:
            return
        # range image projection
        points = self.laser_cloud_in
        x = points["x"]
        y = points["y"]
        z = points["z"]

        dist = np.sqrt(x * x + y * y + z * z)

        valid = (dist >= self.lidar_min_range) & (dist <= self.lidar_max_range)
        if not np.any(valid):
            return

        row_idn = points["ring"].astype(np.int32, copy=False)
        # if sensor is a velodyne (ring_flag == 2), calculate row_idn based on number of scans
        if self.ring_flag == 2:
            xy = np.sqrt(x * x + y * y)
            vertical_angle = np.degrees(np.arctan2(z, xy))
            row_idn = ((vertical_angle + (self.n_scan - 1)) / 2.0).astype(np.int32)

        valid &= (row_idn >= 0) & (row_idn < self.n_scan)
        if not np.any(valid):
            return

        valid &= row_idn % self.downsample_rate == 0
        if not np.any(valid):
            return

        indices = np.flatnonzero(valid)
        rows = row_idn[indices]
        ranges = dist[indices]

        if self.sensor == SensorType.VELODYNE or self.sensor == SensorType.OUSTER:
            horizon_angle = np.arctan2(x[indices], y[indices])
            ang_res_x = 2.0 * np.pi / float(self.horizon_scan)
            cols = (
                -np.rint((horizon_angle - np.pi / 2.0) / ang_res_x) + self.horizon_scan / 2
            ).astype(np.int32)
            cols[cols >= self.horizon_scan] -= self.horizon_scan

        elif self.sensor == SensorType.LIVOX or self.sensor == SensorType.LIVOX360:
            cols = np.empty(len(indices), dtype=np.int32)

            unique_rows = np.unique(rows)
            for row in unique_rows:
                row_mask = rows == row

                count = np.count_nonzero(row_mask)

                start_column = self.column_idn_count_vec[row]

                cols[row_mask] = np.arange(count, dtype=np.int32) + start_column
                self.column_idn_count_vec[row] += count

        else:
            return

        inside = (cols >= 0) & (cols < self.horizon_scan)
        if not np.any(inside):
            return

        indices = indices[inside]
        rows = rows[inside]
        cols = cols[inside]
        ranges = ranges[inside]

        empty = self.range_mat[rows, cols] == np.finfo(np.float32).max
        if not np.any(empty):
            return

        indices = indices[empty]
        rows = rows[empty]
        cols = cols[empty]
        ranges = ranges[empty]

        flat_indices = rows * self.horizon_scan + cols
        _, first = np.unique(flat_indices, return_index=True)
        first.sort()

        indices = indices[first]
        rows = rows[first]
        cols = cols[first]
        ranges = ranges[first]

        selected_points = points[indices]

        if self.deskew_flag != -1 and self.cloud_info.imu_available:
            rel_times = selected_points["time"].astype(np.float64)
            selected_points = self.deskew_points(selected_points, rel_times)

        self.range_mat[rows, cols] = ranges

        full_indices = cols + rows * self.horizon_scan
        self.full_cloud[full_indices] = selected_points

    def cloud_extraction(self) -> None:
        range_mat = self.range_mat
        full_cloud = self.full_cloud

        valid = range_mat != np.finfo(np.float32).max

        flat_indices = np.flatnonzero(valid)
        if flat_indices.size == 0:
            self.extracted_cloud = np.empty(0, dtype=point_xyzirt)
            self.cloud_info.point_col_ind = []
            self.cloud_info.point_range = []
            return

        # extract segmented cloud for lidar odometry
        rows = flat_indices // self.horizon_scan
        cols = flat_indices % self.horizon_scan
        count = flat_indices.size

        # mark the points' column index for marking occlusion later
        self.cloud_info.point_col_ind = cols.tolist()
        # save range info
        self.cloud_info.point_range = range_mat[rows, cols].tolist()
        # size of extracted cloud
        row_counts = np.bincount(rows, minlength=self.n_scan)

        cumulative_counts = np.cumsum(row_counts)
        start_counts = cumulative_counts - row_counts

        self.cloud_info.start_ring_index = (start_counts - 1 + 5).tolist()
        self.cloud_info.end_ring_index = (cumulative_counts - 1 - 5).tolist()

        # save extracted cloud
        self.extracted_cloud = full_cloud[flat_indices].copy()

    def publish_clouds(self) -> None:
        self.cloud_info.header = self.cloud_header
        self.cloud_info.cloud_deskewed = publish_cloud(
            self.pub_extracted_cloud,
            self.extracted_cloud,
            self.cloud_header.stamp,
            self.lidar_frame,
        )
        self.pub_laser_cloud_info.publish(self.cloud_info)


def main(args=None):
    rclpy.init(args=args)

    ip = ImageProjection()

    exec = rclpy.executors.MultiThreadedExecutor()
    exec.add_node(ip)

    ip.get_logger().info("\033[1;32m----> Image Projection Started.\033[0m")

    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        ip.destroy_node()
        rclpy.shutdown()
