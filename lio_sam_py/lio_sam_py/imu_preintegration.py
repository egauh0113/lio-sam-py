# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Copyright 2020 Tixiao Shan
# Copyright 2021 Christoph Gruber
# Copyright 2026 Eugene Auh
# -----------------------------------------------------------------------------
# Original File: src/imuPreintegration.cpp
# Original Author: Tixiao Shan
# Original Source: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
#
# Modifications: This file is a Python port of the original C++ implementation
#                by Eugene Auh in 2026.
# -----------------------------------------------------------------------------
from collections import deque
from copy import deepcopy
import threading

import gtsam
from gtsam.symbol_shorthand import B  # Bias  (ax,ay,az,gx,gy,gz)
from gtsam.symbol_shorthand import V  # Vel   (xdot,ydot,zdot)
from gtsam.symbol_shorthand import X  # Pose3 (x,y,z,r,p,y)
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu

from lio_sam_py.transform_utils import (
    matrix_to_pose,
    matrix_to_transform,
    transform_to_matrix,
    odom_to_pose3,
)
from lio_sam_py.utility import ParamServer, stamp_to_sec, qos, qos_imu


class TransformFusion(ParamServer):
    def __init__(self):
        super().__init__(node_name="transform_fusion")

        self.mtx = threading.Lock()

        self.callback_group_imu_odometry = MutuallyExclusiveCallbackGroup()
        self.sub_imu_odometry = self.create_subscription(
            Odometry,
            self.odom_topic + "_incremental",
            self.imu_odometry_handler,
            qos_imu,
            callback_group=self.callback_group_imu_odometry,
        )
        self.callback_group_laser_odometry = MutuallyExclusiveCallbackGroup()
        self.sub_laser_odometry = self.create_subscription(
            Odometry,
            "lio_sam/mapping/odometry",
            self.lidar_odometry_handler,
            qos,
            callback_group=self.callback_group_laser_odometry,
        )

        self.pub_imu_odometry = self.create_publisher(
            Odometry, self.odom_topic, qos_imu
        )
        self.pub_imu_path = self.create_publisher(Path, "lio_sam/imu/path", qos)

        self.lidar_odom_affine = np.eye(4, dtype=np.float64)

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=10.0)
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.lidar_to_baselink = None

        self.lidar_odom_time = -1
        self.imu_odom_queue = deque()

        self.imu_path = Path()
        self.last_path_time = -1

    @staticmethod
    def odom_to_matrix(odom: Odometry) -> np.ndarray:
        trans = np.array(
            [
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        quat = np.array(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ],
            dtype=np.float64,
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = trans

        return T

    def lidar_odometry_handler(self, odom_msg: Odometry):
        with self.mtx:
            self.lidar_odom_affine = TransformFusion.odom_to_matrix(odom_msg)

            self.lidar_odom_time = stamp_to_sec(odom_msg.header.stamp)

    def imu_odometry_handler(self, odom_msg: Odometry):
        with self.mtx:
            self.imu_odom_queue.append(odom_msg)

            # get latest odometry (at current IMU stamp)
            if self.lidar_odom_time == -1:
                return
            while self.imu_odom_queue:
                if (
                    stamp_to_sec(self.imu_odom_queue[0].header.stamp)
                    <= self.lidar_odom_time
                ):
                    self.imu_odom_queue.popleft()
                else:
                    break
            if not self.imu_odom_queue:
                return
            imu_odom_affine_front = TransformFusion.odom_to_matrix(
                self.imu_odom_queue[0]
            )
            imu_odom_affine_back = TransformFusion.odom_to_matrix(
                self.imu_odom_queue[-1]
            )
            imu_odom_affine_incre = (
                np.linalg.inv(imu_odom_affine_front) @ imu_odom_affine_back
            )
            imu_odom_affine_last = self.lidar_odom_affine @ imu_odom_affine_incre
            t_cur = imu_odom_affine_last.copy()

            # publish latest odometry
            laser_odometry = deepcopy(self.imu_odom_queue[-1])
            matrix_to_pose(t_cur, laser_odometry.pose.pose)
            self.pub_imu_odometry.publish(laser_odometry)

            # publish tf
            if self.lidar_frame != self.baselink_frame:
                try:
                    tf_msg = self.tf_buffer.lookup_transform(
                        self.lidar_frame, self.baselink_frame, rclpy.time.Time()
                    )
                    self.lidar_to_baselink = transform_to_matrix(tf_msg.transform)
                except tf2_ros.TransformException as e:
                    self.get_logger().error(str(e))
                if self.lidar_to_baselink is None:
                    return
                t_cur = t_cur @ self.lidar_to_baselink
            ts = TransformStamped()
            ts.header.stamp = odom_msg.header.stamp
            ts.header.frame_id = self.odometry_frame
            ts.child_frame_id = self.baselink_frame
            matrix_to_transform(t_cur, ts.transform)
            self.tf_broadcaster.sendTransform(ts)

            # publish IMU path
            imu_time = stamp_to_sec(self.imu_odom_queue[-1].header.stamp)
            if imu_time - self.last_path_time > 0.1:
                self.last_path_time = imu_time
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.imu_odom_queue[-1].header.stamp
                pose_stamped.header.frame_id = self.odometry_frame
                pose_stamped.pose = laser_odometry.pose.pose
                self.imu_path.poses.append(pose_stamped)
                while (
                    self.imu_path.poses
                    and stamp_to_sec(self.imu_path.poses[0].header.stamp)
                    < self.lidar_odom_time - 1.0
                ):
                    self.imu_path.poses.pop(0)
                if self.pub_imu_path.get_subscription_count() != 0:
                    self.imu_path.header.stamp = self.imu_odom_queue[-1].header.stamp
                    self.imu_path.header.frame_id = self.odometry_frame
                    self.pub_imu_path.publish(self.imu_path)


class IMUPreintegration(ParamServer):
    def __init__(self):
        super().__init__(node_name="imu_preintegration")

        self.mtx = threading.Lock()

        self.callback_group_imu = MutuallyExclusiveCallbackGroup()
        self.sub_imu = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_handler,
            qos_imu,
            callback_group=self.callback_group_imu,
        )
        self.callback_group_odom = MutuallyExclusiveCallbackGroup()
        self.sub_odometry = self.create_subscription(
            Odometry,
            "lio_sam/mapping/odometry_incremental",
            self.odometry_handler,
            qos,
            callback_group=self.callback_group_odom,
        )
        self.pub_imu_odometry = self.create_publisher(
            Odometry, self.odom_topic + "_incremental", qos_imu
        )

        self.system_initialized = False

        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], dtype=np.float64)
        )  # rad, rad, rad, m, m, m
        self.prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e4)  # m/s
        self.prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, 1e-3
        )  # 1e-2 ~ 1e-3 seems to be good
        self.correction_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1], dtype=np.float64)
        )  # rad, rad, rad, m, m, m
        self.correction_noise2 = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
        )  # rad, rad, rad, m, m, m
        self.noise_model_between_bias = np.array(
            [
                self.imu_acc_bias_n,
                self.imu_acc_bias_n,
                self.imu_acc_bias_n,
                self.imu_gyr_bias_n,
                self.imu_gyr_bias_n,
                self.imu_gyr_bias_n,
            ],
            dtype=np.float64,
        )

        p = gtsam.PreintegrationParams.MakeSharedU(self.imu_gravity)
        p.setAccelerometerCovariance(
            np.identity(3, dtype=np.float64) * self.imu_acc_noise**2
        )  # acc white noise in continuous
        p.setGyroscopeCovariance(
            np.identity(3, dtype=np.float64) * self.imu_gyr_noise**2
        )  # gyro white noise in continuous
        p.setIntegrationCovariance(
            np.identity(3, dtype=np.float64) * (1e-4) ** 2
        )  # error committed in integrating position from velocities
        prior_imu_bias = gtsam.imuBias.ConstantBias(
            np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        )  # assume zero initial bias

        self.imu_integrator_opt = gtsam.PreintegratedImuMeasurements(
            p, prior_imu_bias
        )  # setting up the IMU integration for IMU message thread
        self.imu_integrator_imu = gtsam.PreintegratedImuMeasurements(
            p, prior_imu_bias
        )  # setting up the IMU integration for optimization

        self.imu_que_opt = deque()
        self.imu_que_imu = deque()

        self.prev_pose = gtsam.Pose3()
        self.prev_vel = np.zeros(3, dtype=np.float64)
        self.prev_state = gtsam.NavState(self.prev_pose, self.prev_vel)
        self.prev_bias = gtsam.imuBias.ConstantBias()

        self.prev_state_odom = gtsam.NavState()
        self.prev_bias_odom = gtsam.imuBias.ConstantBias()

        self.done_first_opt = False
        self.last_imu_t_imu = -1
        self.last_imu_t_opt = -1

        self.optimizer = gtsam.ISAM2()
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_values = gtsam.Values()

        self.delta_t = 0

        self.key = 1

        self.imu_to_lidar = gtsam.Pose3(
            gtsam.Rot3(np.eye(3, dtype=np.float64)),
            gtsam.Point3(
                -float(self.ext_trans[0]),
                -float(self.ext_trans[1]),
                -float(self.ext_trans[2]),
            ),
        )
        self.lidar_to_imu = gtsam.Pose3(
            gtsam.Rot3(np.eye(3, dtype=np.float64)),
            gtsam.Point3(
                float(self.ext_trans[0]),
                float(self.ext_trans[1]),
                float(self.ext_trans[2]),
            ),
        )

    def reset_optimization(self):
        opt_parameters = gtsam.ISAM2Params()
        opt_parameters.setRelinearizeThreshold(0.1)
        opt_parameters.relinearizeSkip = 1
        self.optimizer = gtsam.ISAM2(opt_parameters)

        new_graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_factors = new_graph_factors

        new_graph_values = gtsam.Values()
        self.graph_values = new_graph_values

    def reset_params(self):
        self.last_imu_t_imu = -1
        self.done_first_opt = False
        self.system_initialized = False

    def odometry_handler(self, odom_msg: Odometry):
        with self.mtx:
            current_correction_time = stamp_to_sec(odom_msg.header.stamp)

            # make sure we have imu data to integrate
            if not self.imu_que_opt:
                return

            degenerate = int(odom_msg.pose.covariance[0]) == 1
            lidar_pose = odom_to_pose3(odom_msg)

            # 0. initialize system
            if not self.system_initialized:
                self.reset_optimization()

                # pop old IMU message
                while self.imu_que_opt:
                    if (
                        stamp_to_sec(self.imu_que_opt[0].header.stamp)
                        < current_correction_time - self.delta_t
                    ):
                        self.last_imu_t_opt = stamp_to_sec(
                            self.imu_que_opt[0].header.stamp
                        )
                        self.imu_que_opt.popleft()
                    else:
                        break
                # initial pose
                self.prev_pose = lidar_pose.compose(self.lidar_to_imu)
                prior_pose = gtsam.PriorFactorPose3(
                    X(0), self.prev_pose, self.prior_pose_noise
                )
                self.graph_factors.add(prior_pose)
                # initial velocity
                self.prev_vel = np.zeros(3, dtype=np.float64)
                prior_vel = gtsam.PriorFactorVector(
                    V(0), self.prev_vel, self.prior_vel_noise
                )
                self.graph_factors.add(prior_vel)
                # initial bias
                self.prev_bias = gtsam.imuBias.ConstantBias()
                prior_bias = gtsam.PriorFactorConstantBias(
                    B(0), self.prev_bias, self.prior_bias_noise
                )
                self.graph_factors.add(prior_bias)
                # add values
                self.graph_values.insert(X(0), self.prev_pose)
                self.graph_values.insert(V(0), self.prev_vel)
                self.graph_values.insert(B(0), self.prev_bias)
                # optimize once
                self.optimizer.update(self.graph_factors, self.graph_values)
                self.graph_factors.resize(0)
                self.graph_values.clear()

                self.imu_integrator_imu.resetIntegrationAndSetBias(self.prev_bias)
                self.imu_integrator_opt.resetIntegrationAndSetBias(self.prev_bias)

                self.key = 1
                self.system_initialized = True
                return

            # reset graph for speed
            if self.key == 100:
                # get updated noise before reset
                updated_pose_noise = gtsam.noiseModel.Gaussian.Covariance(
                    self.optimizer.marginalCovariance(X(self.key - 1))
                )
                updated_vel_noise = gtsam.noiseModel.Gaussian.Covariance(
                    self.optimizer.marginalCovariance(V(self.key - 1))
                )
                updated_bias_noise = gtsam.noiseModel.Gaussian.Covariance(
                    self.optimizer.marginalCovariance(B(self.key - 1))
                )
                # reset graph
                self.reset_optimization()
                # add pose
                prior_pose = gtsam.PriorFactorPose3(
                    X(0), self.prev_pose, updated_pose_noise
                )
                self.graph_factors.add(prior_pose)
                # add velocity
                prior_vel = gtsam.PriorFactorVector(
                    V(0), self.prev_vel, updated_vel_noise
                )
                self.graph_factors.add(prior_vel)
                # add bias
                prior_bias = gtsam.PriorFactorConstantBias(
                    B(0), self.prev_bias, updated_bias_noise
                )
                self.graph_factors.add(prior_bias)
                # add values
                self.graph_values.insert(X(0), self.prev_pose)
                self.graph_values.insert(V(0), self.prev_vel)
                self.graph_values.insert(B(0), self.prev_bias)
                # optimize once
                self.optimizer.update(self.graph_factors, self.graph_values)
                self.graph_factors.resize(0)
                self.graph_values.clear()

                self.key = 1

            # 1. integrate imu data and optimize
            while self.imu_que_opt:
                # pop and integrate imu data that is between two optimizations
                this_imu = self.imu_que_opt[0]
                imu_time = stamp_to_sec(this_imu.header.stamp)
                if imu_time < current_correction_time - self.delta_t:
                    dt = (
                        1.0 / 500.0
                        if self.last_imu_t_opt < 0
                        else imu_time - self.last_imu_t_opt
                    )
                    self.imu_integrator_opt.integrateMeasurement(
                        np.array(
                            [
                                this_imu.linear_acceleration.x,
                                this_imu.linear_acceleration.y,
                                this_imu.linear_acceleration.z,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                this_imu.angular_velocity.x,
                                this_imu.angular_velocity.y,
                                this_imu.angular_velocity.z,
                            ],
                            dtype=np.float64,
                        ),
                        dt,
                    )
                    self.last_imu_t_opt = imu_time
                    self.imu_que_opt.popleft()
                else:
                    break
            # add imu factor to graph
            imu_factor = gtsam.ImuFactor(
                X(self.key - 1),
                V(self.key - 1),
                X(self.key),
                V(self.key),
                B(self.key - 1),
                self.imu_integrator_opt,
            )
            self.graph_factors.add(imu_factor)
            # add imu bias between factor
            self.graph_factors.add(
                gtsam.BetweenFactorConstantBias(
                    B(self.key - 1),
                    B(self.key),
                    gtsam.imuBias.ConstantBias(),
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.sqrt(self.imu_integrator_opt.deltaTij())
                        * self.noise_model_between_bias
                    ),
                )
            )
            # add pose factor
            cur_pose = lidar_pose.compose(self.lidar_to_imu)
            pose_factor = gtsam.PriorFactorPose3(
                X(self.key),
                cur_pose,
                self.correction_noise2 if degenerate else self.correction_noise,
            )
            self.graph_factors.add(pose_factor)
            # insert predicted values
            prop_state = self.imu_integrator_opt.predict(
                self.prev_state, self.prev_bias
            )
            self.graph_values.insert(X(self.key), prop_state.pose())
            self.graph_values.insert(V(self.key), prop_state.velocity())
            self.graph_values.insert(B(self.key), self.prev_bias)
            # optimize
            self.optimizer.update(self.graph_factors, self.graph_values)
            self.optimizer.update()
            self.graph_factors.resize(0)
            self.graph_values.clear()
            # Overwrite the beginning of the preintegration for the next step.
            result = self.optimizer.calculateEstimate()
            self.prev_pose = result.atPose3(X(self.key))
            self.prev_vel = result.atVector(V(self.key))
            self.prev_state = gtsam.NavState(self.prev_pose, self.prev_vel)
            self.prev_bias = result.atConstantBias(B(self.key))
            # Reset the optimization preintegration object.
            self.imu_integrator_opt.resetIntegrationAndSetBias(self.prev_bias)
            # check optimization
            if self.failure_detection(self.prev_vel, self.prev_bias):
                self.reset_params()
                return

            # 2.  after optimization, re-propagate imu odometry preintegration
            self.prev_state_odom = self.prev_state
            self.prev_bias_odom = self.prev_bias
            # first pop imu message older than current correction data
            last_imu_qt = -1
            while (
                self.imu_que_imu
                and stamp_to_sec(self.imu_que_imu[0].header.stamp)
                < current_correction_time - self.delta_t
            ):
                last_imu_qt = stamp_to_sec(self.imu_que_imu[0].header.stamp)
                self.imu_que_imu.popleft()
            # repropagate
            if self.imu_que_imu:
                # reset bias use the newly opyimized bias
                self.imu_integrator_imu.resetIntegrationAndSetBias(self.prev_bias_odom)
                # integrate imu message from the beginning of this optimization
                for i in range(len(self.imu_que_imu)):
                    this_imu = self.imu_que_imu[i]
                    imu_time = stamp_to_sec(this_imu.header.stamp)
                    dt = 1.0 / 500.0 if last_imu_qt < 0 else imu_time - last_imu_qt

                    self.imu_integrator_imu.integrateMeasurement(
                        np.array(
                            [
                                this_imu.linear_acceleration.x,
                                this_imu.linear_acceleration.y,
                                this_imu.linear_acceleration.z,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                this_imu.angular_velocity.x,
                                this_imu.angular_velocity.y,
                                this_imu.angular_velocity.z,
                            ],
                            dtype=np.float64,
                        ),
                        dt,
                    )
                    last_imu_qt = imu_time

            self.key += 1
            self.done_first_opt = True

    def failure_detection(
        self, vel_cur: np.ndarray, bias_cur: gtsam.imuBias.ConstantBias
    ) -> bool:
        vel = np.asarray(vel_cur, dtype=np.float64)
        if np.linalg.norm(vel) > 30:
            self.get_logger().warn("Large velocity, reset IMU-preintegration!")
            return True

        ba = np.asarray(bias_cur.accelerometer(), dtype=np.float64)
        bg = np.asarray(bias_cur.gyroscope(), dtype=np.float64)
        if np.linalg.norm(ba) > 1.0 or np.linalg.norm(bg) > 1.0:
            self.get_logger().warn("Large bias, reset IMU-preintegration!")
            return True

        return False

    def imu_handler(self, imu_raw: Imu) -> None:
        with self.mtx:
            this_imu = self.imu_converter(imu_raw)

            self.imu_que_opt.append(this_imu)
            self.imu_que_imu.append(this_imu)

            if not self.done_first_opt:
                return

            imu_time = stamp_to_sec(this_imu.header.stamp)
            dt = (
                1.0 / 500.0
                if self.last_imu_t_imu < 0
                else imu_time - self.last_imu_t_imu
            )
            self.last_imu_t_imu = imu_time

            # integrate this single imu message
            self.imu_integrator_imu.integrateMeasurement(
                np.array(
                    [
                        this_imu.linear_acceleration.x,
                        this_imu.linear_acceleration.y,
                        this_imu.linear_acceleration.z,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        this_imu.angular_velocity.x,
                        this_imu.angular_velocity.y,
                        this_imu.angular_velocity.z,
                    ],
                    dtype=np.float64,
                ),
                dt,
            )

            # predict odometry
            current_state = self.imu_integrator_imu.predict(
                self.prev_state_odom, self.prev_bias_odom
            )

            # publish odometry
            odometry = Odometry()
            odometry.header.stamp = this_imu.header.stamp
            odometry.header.frame_id = self.odometry_frame
            odometry.child_frame_id = "odom_imu"

            # transform imu pose to lidar
            imu_pose = gtsam.Pose3(
                current_state.pose().rotation(), current_state.pose().translation()
            )
            lidar_pose = imu_pose.compose(self.imu_to_lidar)

            odometry.pose.pose.position.x = lidar_pose.translation()[0]
            odometry.pose.pose.position.y = lidar_pose.translation()[1]
            odometry.pose.pose.position.z = lidar_pose.translation()[2]
            odometry.pose.pose.orientation.x = lidar_pose.rotation().toQuaternion().x()
            odometry.pose.pose.orientation.y = lidar_pose.rotation().toQuaternion().y()
            odometry.pose.pose.orientation.z = lidar_pose.rotation().toQuaternion().z()
            odometry.pose.pose.orientation.w = lidar_pose.rotation().toQuaternion().w()

            odometry.twist.twist.linear.x = current_state.velocity()[0]
            odometry.twist.twist.linear.y = current_state.velocity()[1]
            odometry.twist.twist.linear.z = current_state.velocity()[2]
            odometry.twist.twist.angular.x = (
                this_imu.angular_velocity.x + self.prev_bias_odom.gyroscope()[0]
            )
            odometry.twist.twist.angular.y = (
                this_imu.angular_velocity.y + self.prev_bias_odom.gyroscope()[1]
            )
            odometry.twist.twist.angular.z = (
                this_imu.angular_velocity.z + self.prev_bias_odom.gyroscope()[2]
            )
            self.pub_imu_odometry.publish(odometry)


def main(args=None):
    rclpy.init(args=args)

    imup = IMUPreintegration()
    tf = TransformFusion()

    exec = rclpy.executors.MultiThreadedExecutor()
    exec.add_node(imup)
    exec.add_node(tf)

    imup.get_logger().info("\033[1;32m----> IMU Preintegration Started.\033[0m")

    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        imup.destroy_node()
        tf.destroy_node()
        rclpy.shutdown()
