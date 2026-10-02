# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2020 Tixiao Shan
# Copyright 2021 Christoph Gruber
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
# Original File: src/featureExtraction.cpp
# Original Author: Tixiao Shan
# Original Source: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
#
# Modifications: This file is a Python port of the original C++ implementation
#                by Eugene Auh in 2026.
# -----------------------------------------------------------------------------
from copy import deepcopy

import numpy as np

import rclpy
from sensor_msgs.msg import PointCloud2

from lio_sam_py.transform_utils import voxel_downsample
from lio_sam_py.utility import (
    POINT_DTYPE,
    ParamServer,
    pointcloud2_to_numpy,
    publish_cloud,
    qos,
)
from lio_sam_msgs.msg import CloudInfo


class FeatureExtraction(ParamServer):
    def __init__(self):
        super().__init__(node_name="feature_extraction")

        self.sub_laser_cloud_info = self.create_subscription(
            CloudInfo, "lio_sam/deskew/cloud_info", self.laser_cloud_info_handler, qos
        )

        self.pub_laser_cloud_info = self.create_publisher(
            CloudInfo, "lio_sam/feature/cloud_info", qos
        )
        self.pub_corner_points = self.create_publisher(
            PointCloud2, "lio_sam/feature/cloud_corner", 1
        )
        self.pub_surface_points = self.create_publisher(
            PointCloud2, "lio_sam/feature/cloud_surface", 1
        )

        self.extracted_cloud = np.empty(0, dtype=POINT_DTYPE)
        self.corner_cloud = np.empty(0, dtype=POINT_DTYPE)
        self.surface_cloud = np.empty(0, dtype=POINT_DTYPE)

        self.cloud_info = CloudInfo()

        self.cloud_smoothness = np.zeros(
            self.n_scan * self.horizon_scan, dtype=np.float32
        )
        self.cloud_curvature = np.zeros(
            self.n_scan * self.horizon_scan, dtype=np.float32
        )
        self.cloud_neighbor_picked = np.zeros(
            self.n_scan * self.horizon_scan, dtype=np.int8
        )
        self.cloud_label = np.zeros(self.n_scan * self.horizon_scan, dtype=np.int8)

    def laser_cloud_info_handler(self, msg_in: CloudInfo) -> None:
        self.cloud_info = deepcopy(msg_in)  # new cloud info
        self.extracted_cloud = pointcloud2_to_numpy(self.cloud_info.cloud_deskewed)

        self.calculate_smoothness()

        self.mark_occluded_points()

        self.extract_features()

        self.publish_feature_cloud()

    def calculate_smoothness(self) -> None:
        self.cloud_smoothness.fill(0)
        self.cloud_curvature.fill(0)
        self.cloud_neighbor_picked.fill(0)
        self.cloud_label.fill(0)

        cloud_size = len(self.extracted_cloud)
        point_range = np.asarray(
            self.cloud_info.point_range[:cloud_size],
            dtype=np.float32,
        )
        if cloud_size < 11:
            return
        diff_range = (
            point_range[:-10]
            + point_range[1:-9]
            + point_range[2:-8]
            + point_range[3:-7]
            + point_range[4:-6]
            - 10 * point_range[5:-5]
            + point_range[6:-4]
            + point_range[7:-3]
            + point_range[8:-2]
            + point_range[9:-1]
            + point_range[10:]
        )

        curvature = diff_range * diff_range
        self.cloud_curvature[5: cloud_size - 5] = curvature

        # cloud_smoothness for sorting
        self.cloud_smoothness[5: cloud_size - 5] = curvature

    def mark_occluded_points(self) -> None:
        cloud_size = len(self.extracted_cloud)
        point_range = np.asarray(
            self.cloud_info.point_range[:cloud_size],
            dtype=np.float32,
        )
        point_col_ind = np.asarray(
            self.cloud_info.point_col_ind[:cloud_size],
            dtype=np.int32,
        )
        # mark occluded points and parallel beam points
        for i in range(5, cloud_size - 6):
            # occluded points
            depth1 = point_range[i]
            depth2 = point_range[i + 1]
            column_diff = abs(point_col_ind[i + 1] - point_col_ind[i])
            if column_diff < 10:
                # 10 pixel diff in range image
                if depth1 - depth2 > 0.3:
                    self.cloud_neighbor_picked[i - 5: i + 1] = 1
                elif depth2 - depth1 > 0.3:
                    self.cloud_neighbor_picked[i + 1: i + 7] = 1
            # parallel beam
            diff1 = abs(point_range[i - 1] - point_range[i])
            diff2 = abs(point_range[i + 1] - point_range[i])

            if diff1 > 0.02 * point_range[i] and diff2 > 0.02 * point_range[i]:
                self.cloud_neighbor_picked[i] = 1

    def extract_features(self) -> None:
        corner_points = []
        surface_points = []

        point_col_ind = np.asarray(
            self.cloud_info.point_col_ind[: len(self.extracted_cloud)],
            dtype=np.int32,
        )
        for i in range(self.n_scan):
            surface_cloud_scan = []

            for j in range(6):
                sp = (
                    self.cloud_info.start_ring_index[i] * (6 - j)
                    + self.cloud_info.end_ring_index[i] * j
                ) // 6
                ep = (
                    self.cloud_info.start_ring_index[i] * (5 - j)
                    + self.cloud_info.end_ring_index[i] * (j + 1)
                ) // 6 - 1

                if sp >= ep:
                    continue

                indices = np.arange(sp, ep + 1, dtype=np.int32)
                indices = indices[
                    np.argsort(self.cloud_smoothness[indices], kind="quicksort")
                ]

                largest_picked_num = 0
                for k in range(len(indices) - 1, -1, -1):
                    ind = int(indices[k])
                    if (
                        self.cloud_neighbor_picked[ind] == 0
                        and self.cloud_curvature[ind] > self.edge_threshold
                    ):
                        largest_picked_num += 1
                        if largest_picked_num <= 20:
                            self.cloud_label[ind] = 1
                            corner_points.append(self.extracted_cloud[ind])
                        else:
                            break

                        self.cloud_neighbor_picked[ind] = 1
                        for m in range(1, 6):
                            column_diff = abs(
                                point_col_ind[ind + m] - point_col_ind[ind + m - 1]
                            )
                            if column_diff > 10:
                                break
                            self.cloud_neighbor_picked[ind + m] = 1
                        for m in range(-1, -6, -1):
                            column_diff = abs(
                                point_col_ind[ind + m] - point_col_ind[ind + m + 1]
                            )
                            if column_diff > 10:
                                break
                            self.cloud_neighbor_picked[ind + m] = 1

                for ind in indices:
                    ind = int(ind)
                    if (
                        self.cloud_neighbor_picked[ind] == 0
                        and self.cloud_curvature[ind] < self.surf_threshold
                    ):
                        self.cloud_label[ind] = -1
                        self.cloud_neighbor_picked[ind] = 1

                        for m in range(1, 6):
                            column_diff = abs(
                                point_col_ind[ind + m] - point_col_ind[ind + m - 1]
                            )
                            if column_diff > 10:
                                break

                            self.cloud_neighbor_picked[ind + m] = 1
                        for m in range(-1, -6, -1):
                            column_diff = abs(
                                point_col_ind[ind + m] - point_col_ind[ind + m + 1]
                            )
                            if column_diff > 10:
                                break

                            self.cloud_neighbor_picked[ind + m] = 1

                for k in range(sp, ep + 1):
                    if self.cloud_label[k] <= 0:
                        surface_cloud_scan.append(self.extracted_cloud[k])

            scan_surface = np.array(surface_cloud_scan, dtype=POINT_DTYPE)
            scan_surface_ds = voxel_downsample(
                scan_surface, self.odometry_surf_leaf_size
            )

            surface_points.append(scan_surface_ds)

        if corner_points:
            self.corner_cloud = np.asarray(corner_points, dtype=POINT_DTYPE)
        else:
            self.corner_cloud = np.empty(0, dtype=POINT_DTYPE)

        if surface_points:
            self.surface_cloud = np.concatenate(surface_points)
        else:
            self.surface_cloud = np.empty(0, dtype=POINT_DTYPE)

    def free_cloud_info_memory(self) -> None:
        self.cloud_info.start_ring_index = []
        self.cloud_info.end_ring_index = []
        self.cloud_info.point_col_ind = []
        self.cloud_info.point_range = []

    def publish_feature_cloud(self) -> None:
        # free cloud info memory
        self.free_cloud_info_memory()
        # save newly extracted features
        self.cloud_info.cloud_corner = publish_cloud(
            self.pub_corner_points,
            self.corner_cloud,
            self.cloud_info.header.stamp,
            self.lidar_frame,
        )
        self.cloud_info.cloud_surface = publish_cloud(
            self.pub_surface_points,
            self.surface_cloud,
            self.cloud_info.header.stamp,
            self.lidar_frame,
        )
        # publish to map_optimization
        self.pub_laser_cloud_info.publish(self.cloud_info)


def main(args=None):
    rclpy.init(args=args)

    fe = FeatureExtraction()

    fe.get_logger().info("\033[1;32m----> Feature Extraction Started.\033[0m")

    try:
        rclpy.spin(fe)
    except KeyboardInterrupt:
        pass
    finally:
        fe.destroy_node()
        rclpy.shutdown()
